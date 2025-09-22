defmodule ReqLLM.Providers.VLLM do
  @moduledoc """
  vLLM provider – 100% OpenAI Chat Completions compatible with high-performance inference.

  ## Protocol Usage

  Uses the generic `ReqLLM.Context.Codec` and `ReqLLM.Response.Codec` protocols.
  No custom wrapper modules – leverages the standard OpenAI-compatible codecs.

  ## vLLM-Specific Extensions

  Beyond standard OpenAI parameters, vLLM supports:

  ### Extra Sampling Parameters
  - `best_of` - Number of completions to generate and return the best one
  - `use_beam_search` - Enable beam search decoding
  - `top_k` - Top-k sampling parameter
  - `min_p` - Minimum probability threshold for token sampling
  - `repetition_penalty` - Penalty for repeating tokens

  ### Guided Decoding
  - `guided_json` - Enforce JSON schema output format
  - `guided_regex` - Constrain output to match regex patterns
  - `guided_choice` - Limit output to specific choice options
  - `guided_grammar` - Follow context-free grammar rules

  See `provider_schema/0` for the complete vLLM-specific schema and
  `ReqLLM.Provider.Options` for inherited OpenAI parameters.

  ## Configuration

  Configure your vLLM server endpoint and optional API key:

      # Add to .env file (automatically loaded)
      VLLM_BASE_URL=http://localhost:8000/v1
      VLLM_API_KEY=token-abc123  # Optional - only if your server requires auth

  The base URL should point to your vLLM server's OpenAI-compatible endpoint.
  Common configurations:
  - Local: `http://localhost:8000/v1`
  - Docker: `http://vllm-server:8000/v1`
  - Remote: `https://your-vllm-server.com/v1`

  ## Examples

      # Simple text generation
      model = ReqLLM.Model.new(:vllm, "your-model-name")
      {:ok, response} = ReqLLM.generate_text(model, "Hello!")

      # With custom server and API key
      opts = [
        provider_options: [
          base_url: "https://your-vllm-server.com/v1",
          api_key: "your-custom-token-here"
        ]
      ]
      {:ok, response} = ReqLLM.generate_text(model, "Hello!", opts)

      # With vLLM-specific sampling
      opts = [
        provider_options: [
          repetition_penalty: 1.1,
          use_beam_search: true,
          min_p: 0.05
        ]
      ]
      {:ok, response} = ReqLLM.generate_text(model, "Tell me a story", opts)

      # Guided JSON output with custom server
      opts = [
        provider_options: [
          base_url: "https://your-vllm-server.com/v1",
          api_key: "your-token",
          guided_json: %{
            "type" => "object",
            "properties" => %{
              "name" => %{"type" => "string"},
              "age" => %{"type" => "integer"}
            }
          }
        ]
      ]
      {:ok, response} = ReqLLM.generate_text(model, "Generate person data", opts)
  """

  @behaviour ReqLLM.Provider

  use ReqLLM.Provider.DSL,
    id: :vllm,
    base_url: "http://localhost:8000/v1",
    metadata: "priv/models_dev/vllm.json",
    default_env_key: "VLLM_API_KEY",
    provider_schema: [
      # Connection configuration
      base_url: [
        type: :string,
        doc: "Custom vLLM server base URL (overrides VLLM_BASE_URL env var)"
      ],
      api_key: [
        type: :string,
        doc: "Custom API key for authentication (overrides VLLM_API_KEY env var)"
      ],
      # Extra sampling parameters
      best_of: [
        type: :pos_integer,
        doc: "Number of completions to generate and return the best one"
      ],
      use_beam_search: [
        type: :boolean,
        doc: "Enable beam search decoding"
      ],
      min_p: [
        type: :float,
        doc: "Minimum probability threshold for token sampling (0.0-1.0)"
      ],
      repetition_penalty: [
        type: :float,
        doc: "Penalty for repeating tokens (typically 1.0-2.0)"
      ],
      # Guided decoding options
      guided_json: [
        type: :map,
        doc: "JSON schema to enforce output format"
      ],
      guided_regex: [
        type: :string,
        doc: "Regex pattern to constrain output"
      ],
      guided_choice: [
        type: {:list, :string},
        doc: "List of specific choice options to limit output"
      ],
      guided_grammar: [
        type: :string,
        doc: "Context-free grammar rules to follow"
      ]
    ]

  import ReqLLM.Provider.Utils, only: [maybe_put: 3]

  require Logger


  @doc """
  Custom prepare_request for :object operations to maintain vLLM-specific structured output handling.

  Uses structured output tools with appropriate token limits for vLLM servers.
  """
  @impl ReqLLM.Provider
  def prepare_request(:object, model_spec, prompt, opts) do
    compiled_schema = Keyword.fetch!(opts, :compiled_schema)

    structured_output_tool =
      ReqLLM.Tool.new!(
        name: "structured_output",
        description: "Generate structured output matching the provided schema",
        parameter_schema: compiled_schema.schema,
        callback: fn _args -> {:ok, "structured output generated"} end
      )

    opts_with_tool =
      opts
      |> Keyword.update(:tools, [structured_output_tool], &[structured_output_tool | &1])
      |> Keyword.put(:tool_choice, %{type: "function", function: %{name: "structured_output"}})
      |> Keyword.put_new(:max_tokens, 4096)

    # Preserve the :object operation for response decoding
    opts_with_operation = Keyword.put(opts_with_tool, :operation, :object)

    prepare_request(:chat, model_spec, prompt, opts_with_operation)
  end

  # Delegate all other operations to defaults with custom base URL handling
  def prepare_request(operation, model_spec, input, opts) do
    case ReqLLM.Provider.Defaults.prepare_request(__MODULE__, operation, model_spec, input, opts) do
      {:ok, request} ->
        # Override base URL if provided in provider_options
        updated_request = maybe_override_base_url(request, opts)
        {:ok, updated_request}

      error ->
        error
    end
  end

  @doc """
  Custom body encoding that adds vLLM-specific extensions to the default OpenAI-compatible format.

  Adds support for vLLM's extra sampling parameters and guided decoding options.
  """
  @impl ReqLLM.Provider
  def encode_body(request) do
    # Handle custom API key authentication
    request = maybe_override_auth_header(request)

    # Start with default encoding
    request = ReqLLM.Provider.Defaults.default_encode_body(request)

    # Parse the encoded body to add vLLM-specific options
    body = Jason.decode!(request.body)

    enhanced_body =
      body
      # Extra sampling parameters
      |> maybe_put(:best_of, request.options[:best_of])
      |> maybe_put(:use_beam_search, request.options[:use_beam_search])
      |> maybe_put(:min_p, request.options[:min_p])
      |> maybe_put(:repetition_penalty, request.options[:repetition_penalty])
      # Guided decoding options
      |> maybe_put(:guided_json, request.options[:guided_json])
      |> maybe_put(:guided_regex, request.options[:guided_regex])
      |> maybe_put(:guided_choice, request.options[:guided_choice])
      |> maybe_put(:guided_grammar, request.options[:guided_grammar])

    # Re-encode with vLLM extensions
    encoded_body = Jason.encode!(enhanced_body)
    Map.put(request, :body, encoded_body)
  end

  # Private helper to override base URL if provided in provider_options
  defp maybe_override_base_url(request, opts) do
    provider_opts = Keyword.get(opts, :provider_options, [])

    case Keyword.get(provider_opts, :base_url) do
      nil ->
        # Check environment variable override
        case System.get_env("VLLM_BASE_URL") do
          nil -> request
          env_url -> update_request_url(request, env_url)
        end

      custom_url ->
        update_request_url(request, custom_url)
    end
  end

  defp update_request_url(request, new_base_url) do
    # Parse the new base URL
    new_uri = URI.parse(new_base_url)

    # Update the request URL while preserving the path
    updated_url = %{request.url |
      scheme: new_uri.scheme,
      host: new_uri.host,
      port: new_uri.port,
      path: Path.join(new_uri.path || "", request.url.path || "")
    }

    %{request | url: updated_url}
  end

  defp maybe_override_auth_header(request) do
    provider_opts = get_provider_options(request.options)

    case Keyword.get(provider_opts, :api_key) do
      nil ->
        request

      custom_api_key ->
        # Override the authorization header
        updated_headers = Map.put(request.headers, "authorization", ["Bearer #{custom_api_key}"])
        %{request | headers: updated_headers}
    end
  end

  # Helper to get provider options from either a map or keyword list
  defp get_provider_options(options) when is_map(options) do
    Map.get(options, :provider_options, [])
  end

  defp get_provider_options(options) when is_list(options) do
    Keyword.get(options, :provider_options, [])
  end
end