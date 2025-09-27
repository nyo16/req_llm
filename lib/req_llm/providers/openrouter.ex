defmodule ReqLLM.Providers.OpenRouter do
  @moduledoc """
  OpenRouter provider – OpenAI Chat Completions compatible with OpenRouter's unified API.

  ## Implementation

  Uses built-in OpenAI-style encoding/decoding defaults.
  No custom wrapper modules – leverages the standard OpenAI-compatible implementations.

  ## OpenRouter-Specific Extensions

  Beyond standard OpenAI parameters, OpenRouter supports:
  - `openrouter_models` - Array of model IDs for routing/fallback preferences
  - `openrouter_route` - Routing strategy (e.g., "fallback")
  - `openrouter_provider` - Provider preferences object for routing decisions
  - `openrouter_transforms` - Array of prompt transforms to apply
  - `openrouter_top_k` - Top-k sampling (not available for OpenAI models)
  - `openrouter_repetition_penalty` - Repetition penalty for reducing repetitive text
  - `openrouter_min_p` - Minimum probability threshold for sampling
  - `openrouter_top_a` - Top-a sampling parameter
  - `app_referer` - HTTP-Referer header for app identification
  - `app_title` - X-Title header for app title in rankings

  ## App Attribution Headers

  OpenRouter supports optional headers for app discoverability:
  - Set `HTTP-Referer` header for app identification
  - Set `X-Title` header for app title in rankings

  See `provider_schema/0` for the complete OpenRouter-specific schema and
  `ReqLLM.Provider.Options` for inherited OpenAI parameters.

  ## Configuration

      # Add to .env file (automatically loaded)
      OPENROUTER_API_KEY=sk-or-...
  """

  @behaviour ReqLLM.Provider

  use ReqLLM.Provider.DSL,
    id: :openrouter,
    base_url: "https://openrouter.ai/api/v1",
    metadata: "priv/models_dev/openrouter.json",
    default_env_key: "OPENROUTER_API_KEY",
    provider_schema: [
      openrouter_models: [
        type: {:list, :string},
        doc: "Array of model IDs for routing/fallback preferences"
      ],
      openrouter_route: [
        type: :string,
        doc: "Routing strategy (e.g., 'fallback')"
      ],
      openrouter_provider: [
        type: :map,
        doc: "Provider preferences object for routing decisions"
      ],
      openrouter_transforms: [
        type: {:list, :string},
        doc: "Array of prompt transforms to apply"
      ],
      openrouter_top_k: [
        type: :integer,
        doc: "Top-k sampling (not available for OpenAI models)"
      ],
      openrouter_repetition_penalty: [
        type: :float,
        doc: "Repetition penalty for reducing repetitive text"
      ],
      openrouter_min_p: [
        type: :float,
        doc: "Minimum probability threshold for sampling"
      ],
      openrouter_top_a: [
        type: :float,
        doc: "Top-a sampling parameter"
      ],
      openrouter_top_logprobs: [
        type: :integer,
        doc: "Number of top log probabilities to return"
      ],
      app_referer: [
        type: :string,
        doc: "HTTP-Referer header for app identification on OpenRouter"
      ],
      app_title: [
        type: :string,
        doc: "X-Title header for app title in OpenRouter rankings"
      ]
    ]

  import ReqLLM.Provider.Utils, only: [maybe_put: 3]

  require Logger

  # Override attach to add app attribution headers
  @impl ReqLLM.Provider
  def attach(request, model_input, user_opts) do
    # Call the default attach implementation first
    request = ReqLLM.Provider.Defaults.default_attach(__MODULE__, request, model_input, user_opts)

    # Add OpenRouter app attribution headers during attach so they're available in tests
    maybe_add_attribution_headers(request, user_opts)
  end

  @doc """
  Custom prepare_request for :object operations to maintain OpenRouter-specific max_tokens handling.

  Ensures that structured output requests have adequate token limits while delegating
  other operations to the default implementation.
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

    # Adjust max_tokens for structured output with OpenRouter-specific minimums
    opts_with_tokens =
      case Keyword.get(opts_with_tool, :max_tokens) do
        nil -> Keyword.put(opts_with_tool, :max_tokens, 4096)
        tokens when tokens < 200 -> Keyword.put(opts_with_tool, :max_tokens, 200)
        _tokens -> opts_with_tool
      end

    # Use the default chat preparation with structured output tools
    ReqLLM.Provider.Defaults.prepare_request(
      __MODULE__,
      :chat,
      model_spec,
      prompt,
      opts_with_tokens
    )
  end

  # Override to reject unsupported operations
  def prepare_request(:embedding, _model_spec, _input, _opts) do
    supported_operations = [:chat, :object]

    {:error,
     ReqLLM.Error.Invalid.Parameter.exception(
       parameter:
         "operation: :embedding not supported by #{inspect(__MODULE__)}. Supported operations: #{inspect(supported_operations)}"
     )}
  end

  # Delegate other operations to default implementation
  def prepare_request(operation, model_spec, input, opts) do
    ReqLLM.Provider.Defaults.prepare_request(__MODULE__, operation, model_spec, input, opts)
  end

  @impl ReqLLM.Provider
  def translate_options(_operation, model, opts) do
    warnings = []

    # Handle legacy parameter names -> OpenRouter prefixed names
    legacy_mappings = [
      {:models, :openrouter_models},
      {:route, :openrouter_route},
      {:provider, :openrouter_provider},
      {:transforms, :openrouter_transforms},
      {:top_k, :openrouter_top_k},
      {:repetition_penalty, :openrouter_repetition_penalty},
      {:min_p, :openrouter_min_p},
      {:top_a, :openrouter_top_a},
      {:top_logprobs, :openrouter_top_logprobs}
    ]

    {opts, warnings} =
      Enum.reduce(legacy_mappings, {opts, warnings}, fn {legacy_key, new_key},
                                                        {acc_opts, acc_warnings} ->
        case Keyword.pop(acc_opts, legacy_key) do
          {nil, remaining_opts} ->
            {remaining_opts, acc_warnings}

          {value, remaining_opts} ->
            warning = "#{legacy_key} is deprecated, use #{new_key} instead"
            {Keyword.put(remaining_opts, new_key, value), [warning | acc_warnings]}
        end
      end)

    # Validate top_k with OpenAI models warning
    {top_k, opts} = Keyword.pop(opts, :openrouter_top_k)

    {opts, warnings} =
      if top_k && String.starts_with?(model.model, "openai/") do
        warning =
          "openrouter_top_k is not available for OpenAI models on OpenRouter and will be ignored"

        {opts, [warning | warnings]}
      else
        opts = if top_k, do: Keyword.put(opts, :openrouter_top_k, top_k), else: opts
        {opts, warnings}
      end

    {opts, Enum.reverse(warnings)}
  end

  @doc """
  Custom body encoding that adds OpenRouter-specific extensions to the default OpenAI-compatible format.

  Adds support for OpenRouter routing and sampling parameters:
  - models (routing preferences)
  - route (routing strategy)
  - provider (provider preferences)
  - transforms (prompt transforms)
  - top_k, repetition_penalty, min_p, top_a (sampling parameters)
  - top_logprobs (log probabilities)

  Also handles OpenRouter-specific app attribution headers:
  - HTTP-Referer header for app identification
  - X-Title header for app title in rankings
  """
  @impl ReqLLM.Provider
  def encode_body(request) do
    # Start with default encoding
    request = ReqLLM.Provider.Defaults.default_encode_body(request)

    # Parse the encoded body to add OpenRouter-specific options
    body = Jason.decode!(request.body)

    enhanced_body =
      body
      |> maybe_put(:models, request.options[:openrouter_models])
      |> maybe_put(:route, request.options[:openrouter_route])
      |> maybe_put(:provider, request.options[:openrouter_provider])
      |> maybe_put(:transforms, request.options[:openrouter_transforms])
      |> maybe_put(:top_k, request.options[:openrouter_top_k])
      |> maybe_put(:repetition_penalty, request.options[:openrouter_repetition_penalty])
      |> maybe_put(:min_p, request.options[:openrouter_min_p])
      |> maybe_put(:top_a, request.options[:openrouter_top_a])
      |> maybe_put(:top_logprobs, request.options[:openrouter_top_logprobs])
      |> add_openrouter_specific_options(request.options)

    # Re-encode with OpenRouter extensions
    encoded_body = Jason.encode!(enhanced_body)
    request = Map.put(request, :body, encoded_body)

    # Add OpenRouter app attribution headers
    request = maybe_add_attribution_headers(request, request.options)

    request
  end

  # Helper function for adding OpenRouter-specific body options not covered by defaults
  defp add_openrouter_specific_options(body, request_options) do
    # Add OpenRouter-specific options that aren't handled by the default encoding
    openrouter_options = [
      # OpenRouter supports this but defaults might not include it
      :logit_bias,
      # OpenRouter supports multiple completions
      :n
    ]

    Enum.reduce(openrouter_options, body, fn key, acc ->
      maybe_put(acc, key, request_options[key])
    end)
  end

  # Helper function for adding OpenRouter app attribution headers
  defp maybe_add_attribution_headers(request, opts) do
    # Get referer from either request options or passed opts
    referer = opts[:app_referer] || request.options[:app_referer]
    title = opts[:app_title] || request.options[:app_title]

    request =
      case referer do
        referer when is_binary(referer) ->
          Req.Request.put_header(request, "HTTP-Referer", referer)

        _ ->
          request
      end

    case title do
      title when is_binary(title) ->
        Req.Request.put_header(request, "X-Title", title)

      _ ->
        request
    end
  end
end
