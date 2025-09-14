defmodule ReqLLM.Providers.CompatibleWithOpenAI do
  @moduledoc """
  OpenAI-compatible endpoint provider for local models and custom servers.

  This provider allows using any OpenAI-compatible endpoint (LM Studio, Ollama, MLX, etc.)
  with flexible model names and configuration. Unlike the main OpenAI provider, this one
  skips model validation and allows custom base URLs and API keys.

  ## Configuration

  This provider is designed for local or custom servers, so API key can be optional:

      # Option 1: No API key (for local servers)
      {:ok, response} = ReqLLM.generate_text(model, "Hello!")

      # Option 2: Pass API key in request options
      {:ok, response} = ReqLLM.generate_text(model, "Hello!", api_key: "custom-key")

      # Option 3: Set via environment (for servers that require auth)
      COMPATIBLE_WITH_OPENAI_API_KEY=your-key

  ## Examples

      # Local MLX server
      model = ReqLLM.Model.from("compatible_with_openai:qwen3-4b-instruct-2507-mlx")
      {:ok, response} = ReqLLM.generate_text(
        model,
        "What day is it today?",
        base_url: "http://localhost:1234/v1",
        temperature: 0.7
      )

      # LM Studio
      model = ReqLLM.Model.from("compatible_with_openai:llama-3.1-8b-instruct")
      {:ok, response} = ReqLLM.generate_text(
        model,
        "Hello!",
        base_url: "http://localhost:1234/v1"
      )

      # Ollama
      model = ReqLLM.Model.from("compatible_with_openai:llama3.1")
      {:ok, response} = ReqLLM.generate_text(
        model,
        "Hello!",
        base_url: "http://localhost:11434/v1"
      )

      # Custom server with authentication
      model = ReqLLM.Model.from("compatible_with_openai:custom-model")
      {:ok, response} = ReqLLM.generate_text(
        model,
        "Hello!",
        base_url: "https://your-server.com/v1",
        api_key: "your-custom-key"
      )

  """

  @behaviour ReqLLM.Provider

  use ReqLLM.Provider.DSL,
    id: :compatible_with_openai,
    base_url: "http://localhost:1234/v1",  # Default to common local server port
    default_env_key: "COMPATIBLE_WITH_OPENAI_API_KEY",
    context_wrapper: ReqLLM.Providers.CompatibleWithOpenAI.Context,
    response_wrapper: ReqLLM.Providers.CompatibleWithOpenAI.Response,
    provider_schema: [
      base_url: [
        type: :string,
        doc: "Custom base URL for OpenAI-compatible endpoint"
      ],
      api_key: [
        type: :string,
        doc: "API key for authentication (optional for local servers)"
      ],
      token: [
        type: :string,
        doc: "Alternative to api_key (optional for local servers)"
      ]
    ]

  import ReqLLM.Provider.Utils,
    only: [prepare_options!: 3, maybe_put: 3, ensure_parsed_body: 1]

  @doc """
  Attaches the CompatibleWithOpenAI plugin to a Req request.

  ## Parameters

    * `request` - The Req request to attach to
    * `model_input` - The model (ReqLLM.Model struct, string, or tuple) that triggers this provider
    * `opts` - Options keyword list (validated against comprehensive schema)

  ## Provider Options (via provider_options key)

    * `:base_url` - Custom base URL for OpenAI-compatible endpoint
    * `:api_key` - API key for authentication (optional for local servers)
    * `:token` - Alternative to api_key (optional for local servers)

  ## Standard Options

    * `:temperature` - Controls randomness (0.0-2.0). Defaults to 0.7
    * `:max_tokens` - Maximum tokens to generate. Defaults to 1024
    * `:system_prompt` - System message to prepend
    * `:tools` - List of tool definitions for function calling
    * All standard ReqLLM generation options are supported

  """
  @impl ReqLLM.Provider
  def prepare_request(:chat, model_input, %ReqLLM.Context{} = context, opts) do
    with {:ok, model} <- ReqLLM.Model.from(model_input) do
      http_opts = Keyword.get(opts, :req_http_options, [])

      request =
        Req.new([url: "/chat/completions", method: :post, receive_timeout: 30_000] ++ http_opts)
        |> attach(model, Keyword.put(opts, :context, context))

      {:ok, request}
    end
  end

  def prepare_request(:embedding, model_input, text, opts) do
    with {:ok, model} <- ReqLLM.Model.from(model_input) do
      http_opts = Keyword.get(opts, :req_http_options, [])

      request =
        Req.new([url: "/embeddings", method: :post, receive_timeout: 30_000] ++ http_opts)
        |> attach(model, Keyword.merge(opts, text: text, operation: :embedding))

      {:ok, request}
    end
  end

  def prepare_request(operation, _model, _input, _opts) do
    {:error,
     ReqLLM.Error.Invalid.Parameter.exception(
       parameter:
         "operation: #{inspect(operation)} not supported by CompatibleWithOpenAI provider. Supported operations: [:chat, :embedding]"
     )}
  end

  @spec attach(Req.Request.t(), ReqLLM.Model.t() | String.t() | {atom(), keyword()}, keyword()) ::
          Req.Request.t()
  @impl ReqLLM.Provider
  def attach(%Req.Request{} = request, model_input, user_opts \\ []) do
    %ReqLLM.Model{} = model = ReqLLM.Model.from!(model_input)

    if model.provider != provider_id() do
      raise ReqLLM.Error.Invalid.Provider.exception(provider: model.provider)
    end

    # No model validation for compatible endpoints - any model name is allowed

    # Extract special keys that shouldn't be validated
    {tools, temp_opts} = Keyword.pop(user_opts, :tools, [])
    {operation, temp_opts} = Keyword.pop(temp_opts, :operation, nil)
    {text, temp_opts} = Keyword.pop(temp_opts, :text, nil)

    # Extract provider-specific options (already validated by dynamic schema)
    provider_opts = Keyword.get(temp_opts, :provider_options, [])

    # Remove provider_options from main opts since we handle them separately
    {_provider_options, core_opts} = Keyword.pop(temp_opts, :provider_options, [])

    # Get API key from provider_options first, then user_opts, then environment (optional)
    api_key = provider_opts[:api_key] || provider_opts[:token] ||
              Keyword.get(user_opts, :api_key) || Keyword.get(user_opts, :token)

    api_key =
      if !api_key || api_key == "" do
        api_key_env = ReqLLM.Provider.Registry.get_env_key(:compatible_with_openai)
        JidoKeys.get(api_key_env) || ""
      else
        api_key
      end

    # Prepare validated core options
    opts = prepare_options!(__MODULE__, model, core_opts)

    # Add tools back after validation
    opts = Keyword.put(opts, :tools, tools)

    # Merge provider-specific options into opts for encoding
    opts = Keyword.merge(opts, provider_opts)

    # Add back special keys
    opts =
      opts
      |> maybe_put(:operation, operation)
      |> maybe_put(:text, text)

    # Get base_url from provider_options first, then user_opts, then default
    base_url = provider_opts[:base_url] || Keyword.get(user_opts, :base_url, default_base_url())
    req_keys = __MODULE__.supported_provider_options() ++ [:model, :context, :operation, :text]

    request_with_options =
      request
      |> Req.Request.register_options(req_keys)
      |> Req.Request.merge_options(Keyword.take(opts, req_keys) ++ [base_url: base_url])

    # Add authorization header only if API key is provided
    request_with_auth =
      if api_key && api_key != "" do
        Req.Request.put_header(request_with_options, "authorization", "Bearer #{api_key}")
      else
        request_with_options
      end

    request_with_auth
    |> ReqLLM.Step.Error.attach()
    |> Req.Request.append_request_steps(llm_encode_body: &__MODULE__.encode_body/1)
    |> ReqLLM.Step.Stream.maybe_attach(opts[:stream])
    |> Req.Request.append_response_steps(llm_decode_response: &__MODULE__.decode_response/1)
    |> ReqLLM.Step.Usage.attach(model)
  end

  @impl ReqLLM.Provider
  def extract_usage(body, _model) when is_map(body) do
    case body do
      %{"usage" => usage} -> {:ok, usage}
      _ -> {:error, :no_usage_found}
    end
  end

  def extract_usage(_, _), do: {:error, :invalid_body}

  # Req pipeline steps - reuse OpenAI's encoding/decoding logic
  @impl ReqLLM.Provider
  def encode_body(request) do
    body =
      case request.options[:operation] do
        :embedding ->
          encode_embedding_body(request)

        _ ->
          encode_chat_body(request)
      end

    try do
      encoded_body = Jason.encode!(body)

      request
      |> Req.Request.put_header("content-type", "application/json")
      |> Map.put(:body, encoded_body)
    rescue
      error ->
        reraise error, __STACKTRACE__
    end
  end

  defp encode_chat_body(request) do
    context_data =
      case request.options[:context] do
        %ReqLLM.Context{} = ctx ->
          ctx
          |> wrap_context()
          |> ReqLLM.Context.Codec.encode_request()

        _ ->
          %{messages: request.options[:messages] || []}
      end

    # Get the model name (compatible endpoints use any model name)
    model = request.options[:model]
    model_name = if is_struct(model, ReqLLM.Model), do: model.model, else: model

    body =
      %{model: model_name}
      |> Map.merge(context_data)
      |> maybe_put(:temperature, request.options[:temperature])
      |> maybe_put(:max_tokens, request.options[:max_tokens])
      |> maybe_put(:top_p, request.options[:top_p])
      |> maybe_put(:stream, request.options[:stream])
      |> maybe_put(:frequency_penalty, request.options[:frequency_penalty])
      |> maybe_put(:presence_penalty, request.options[:presence_penalty])
      |> maybe_put(:stop, request.options[:stop])
      |> maybe_put(:user, request.options[:user])
      |> maybe_put(:seed, request.options[:seed])

    # Handle tools if provided
    body =
      case request.options[:tools] do
        tools when is_list(tools) and tools != [] ->
          body = Map.put(body, :tools, Enum.map(tools, &ReqLLM.Tool.to_schema(&1, :openai)))

          # Handle tool_choice if provided
          case request.options[:tool_choice] do
            nil -> body
            choice -> Map.put(body, :tool_choice, choice)
          end

        _ ->
          body
      end

    # Handle response format if provided
    case request.options[:response_format] do
      format when is_map(format) ->
        Map.put(body, :response_format, format)

      _ ->
        body
    end
  end

  defp encode_embedding_body(request) do
    input = request.options[:text]

    %{
      model: request.options[:model] || request.options[:id],
      input: input
    }
    |> maybe_put(:dimensions, request.options[:dimensions])
    |> maybe_put(:encoding_format, request.options[:encoding_format])
    |> maybe_put(:user, request.options[:user])
  end

  @impl ReqLLM.Provider
  def decode_response({req, resp}) do
    case resp.status do
      200 ->
        body = ensure_parsed_body(resp.body)
        # Return raw parsed data directly - no wrapping needed
        {req, %{resp | body: body}}

      status ->
        err =
          ReqLLM.Error.API.Response.exception(
            reason: "CompatibleWithOpenAI API error",
            status: status,
            response_body: resp.body
          )

        {req, err}
    end
  end
end