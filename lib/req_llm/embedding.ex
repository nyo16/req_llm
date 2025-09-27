defmodule ReqLLM.Embedding do
  @moduledoc """
  Embedding functionality for ReqLLM.

  This module provides embedding generation capabilities with support for:
  - Single text embedding generation
  - Batch text embedding generation
  - Model validation for embedding support

  Currently only OpenAI models are supported for embeddings.
  """

  alias ReqLLM.Model

  # Get embedding models dynamically from metadata
  defp get_embedding_models do
    ReqLLM.Provider.Registry.list_providers()
    |> Enum.flat_map(fn provider ->
      case ReqLLM.Provider.Registry.get_provider_metadata(provider) do
        {:ok, %{models: models}} when is_list(models) ->
          models
          |> Enum.filter(fn model ->
            Map.get(model, "type") == "embedding" or Map.get(model, :type) == "embedding"
          end)
          |> Enum.map(fn model ->
            model_id = Map.get(model, :id) || Map.get(model, "id")
            if model_id, do: "#{provider}:#{model_id}"
          end)
          |> Enum.filter(&(!is_nil(&1)))

        _ ->
          []
      end
    end)
  end

  @base_schema NimbleOptions.new!(
                 dimensions: [
                   type: :pos_integer,
                   doc: "Number of dimensions for the embedding vector"
                 ],
                 encoding_format: [
                   type: {:in, ["float", "base64"]},
                   doc: "Format for encoding the embedding vector",
                   default: "float"
                 ],
                 user: [
                   type: :string,
                   doc: "User identifier for tracking and abuse detection"
                 ],
                 provider_options: [
                   type: {:or, [:map, {:list, :any}]},
                   doc: "Provider-specific options (keyword list or map)",
                   default: []
                 ],
                 req_http_options: [
                   type: {:or, [:map, {:list, :any}]},
                   doc: "Req-specific options (keyword list or map)",
                   default: []
                 ],
                 fixture: [
                   type: {:or, [:string, {:tuple, [:atom, :string]}]},
                   doc: "HTTP fixture for testing (provider inferred from model if string)"
                 ]
               )

  @doc """
  Returns the list of supported embedding model specifications.

  ## Examples

      ReqLLM.Embedding.supported_models()
      #=> ["openai:text-embedding-3-small", "openai:text-embedding-3-large", "openai:text-embedding-ada-002", "google:gemini-embedding-001"]

  """
  @spec supported_models() :: [String.t()]
  def supported_models, do: get_embedding_models()

  @doc """
  Validates that a model supports embedding operations.

  ## Parameters

    * `model_spec` - Model specification in various formats

  ## Examples

      ReqLLM.Embedding.validate_model("openai:text-embedding-3-small")
      #=> {:ok, %ReqLLM.Model{provider: :openai, model: "text-embedding-3-small"}}

      ReqLLM.Embedding.validate_model("anthropic:claude-3-sonnet")
      #=> {:error, :embedding_not_supported}

  """
  @spec validate_model(String.t() | {atom(), keyword()} | struct()) ::
          {:ok, Model.t()} | {:error, term()}
  def validate_model(model_spec) do
    with {:ok, model} <- Model.from(model_spec) do
      model_string = "#{model.provider}:#{model.model}"

      # Check if model is in the embedding models list
      embedding_models = get_embedding_models()

      if model_string in embedding_models do
        # Also verify the provider supports embedding operations
        case ReqLLM.provider(model.provider) do
          {:ok, provider_module} ->
            # Test if provider can prepare embedding request
            case provider_module.prepare_request(:embedding, model, "test", []) do
              {:ok, _} ->
                {:ok, model}

              {:error, _} ->
                {:error,
                 ReqLLM.Error.Invalid.Parameter.exception(
                   parameter:
                     "model: #{model_string} provider does not support embedding operations"
                 )}
            end

          {:error, _} ->
            {:error,
             ReqLLM.Error.Invalid.Parameter.exception(
               parameter: "model: #{model_string} provider not found"
             )}
        end
      else
        {:error,
         ReqLLM.Error.Invalid.Parameter.exception(
           parameter: "model: #{model_string} does not support embedding operations"
         )}
      end
    end
  end

  @doc """
  Returns the base embedding options schema.

  This schema contains embedding-specific options that are vendor-neutral.
  """
  @spec schema :: NimbleOptions.t()
  def schema, do: @base_schema

  @doc """
  Generates embeddings for a single text input.

  ## Parameters

    * `model_spec` - Model specification in various formats
    * `text` - Text to generate embeddings for
    * `opts` - Additional options (keyword list)

  ## Options

    * `:dimensions` - Number of dimensions for embeddings
    * `:encoding_format` - Format for encoding ("float" or "base64")
    * `:user` - User identifier for tracking
    * `:provider_options` - Provider-specific options

  ## Examples

      {:ok, embedding} = ReqLLM.Embedding.embed("openai:text-embedding-3-small", "Hello world")
      #=> {:ok, [0.1, -0.2, 0.3, ...]}

  """
  @spec embed(
          String.t() | {atom(), keyword()} | struct(),
          String.t(),
          keyword()
        ) :: {:ok, [float()]} | {:error, term()}
  def embed(model_spec, text, opts \\ []) do
    with {:ok, model} <- validate_model(model_spec),
         :ok <- validate_text(text),
         {:ok, provider_module} <- ReqLLM.provider(model.provider),
         {:ok, request} <- provider_module.prepare_request(:embedding, model, text, opts),
         {:ok, %Req.Response{status: status, body: decoded_response}} when status in 200..299 <-
           Req.request(request) do
      extract_single_embedding(decoded_response)
    else
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error,
         ReqLLM.Error.API.Request.exception(
           reason: "HTTP #{status}: Request failed",
           status: status,
           response_body: body
         )}

      {:error, error} ->
        {:error, error}
    end
  end

  @doc """
  Generates embeddings for multiple text inputs.

  ## Parameters

    * `model_spec` - Model specification in various formats
    * `texts` - List of texts to generate embeddings for
    * `opts` - Additional options (keyword list)

  ## Options

  Same as `embed/3`.

  ## Examples

      {:ok, embeddings} = ReqLLM.Embedding.embed_many(
        "openai:text-embedding-3-small",
        ["Hello", "World"]
      )
      #=> {:ok, [[0.1, -0.2, ...], [0.3, 0.4, ...]]}

  """
  @spec embed_many(
          String.t() | {atom(), keyword()} | struct(),
          [String.t()],
          keyword()
        ) :: {:ok, [[float()]]} | {:error, term()}
  def embed_many(model_spec, texts, opts \\ []) when is_list(texts) do
    with {:ok, model} <- validate_model(model_spec),
         :ok <- validate_texts(texts),
         {:ok, provider_module} <- ReqLLM.provider(model.provider),
         {:ok, request} <- provider_module.prepare_request(:embedding, model, texts, opts),
         {:ok, %Req.Response{status: status, body: decoded_response}} when status in 200..299 <-
           Req.request(request) do
      extract_multiple_embeddings(decoded_response)
    else
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error,
         ReqLLM.Error.API.Request.exception(
           reason: "HTTP #{status}: Request failed",
           status: status,
           response_body: body
         )}

      {:error, error} ->
        {:error, error}
    end
  end

  defp validate_text("") do
    {:error, ReqLLM.Error.Invalid.Parameter.exception(parameter: "text: cannot be empty")}
  end

  defp validate_text(text) when is_binary(text) do
    :ok
  end

  defp validate_texts([]) do
    {:error, ReqLLM.Error.Invalid.Parameter.exception(parameter: "texts: cannot be empty")}
  end

  defp validate_texts(texts) when is_list(texts) do
    :ok
  end

  defp extract_single_embedding(%{"data" => [%{"embedding" => embedding}]}) do
    {:ok, embedding}
  end

  defp extract_single_embedding(response) do
    {:error,
     ReqLLM.Error.API.Response.exception(
       reason: "Invalid embedding response format",
       response_body: response
     )}
  end

  defp extract_multiple_embeddings(%{"data" => data}) when is_list(data) do
    embeddings =
      data
      |> Enum.sort_by(& &1["index"])
      |> Enum.map(& &1["embedding"])

    {:ok, embeddings}
  end

  defp extract_multiple_embeddings(response) do
    {:error,
     ReqLLM.Error.API.Response.exception(
       reason: "Invalid embedding response format",
       response_body: response
     )}
  end
end
