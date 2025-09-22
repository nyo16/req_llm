defmodule ReqLLM.Providers.VLLMTest do
  @moduledoc """
  Provider-level tests for vLLM implementation.

  Tests the provider contract directly without going through Generation layer.
  Focus: prepare_request -> attach -> request -> decode pipeline.
  """

  use ReqLLM.ProviderCase, provider: ReqLLM.Providers.VLLM

  import ReqLLM.ProviderTestHelpers

  alias ReqLLM.Providers.VLLM

  describe "provider contract" do
    test "provider identity and configuration" do
      assert is_atom(VLLM.provider_id())
      assert is_binary(VLLM.default_base_url())
      assert String.starts_with?(VLLM.default_base_url(), "http")
    end

    test "provider schema separation from core options" do
      schema_keys = VLLM.provider_schema().schema |> Keyword.keys()
      core_keys = ReqLLM.Provider.Options.generation_schema().schema |> Keyword.keys()

      # Provider-specific keys should not overlap with core generation keys
      overlap = MapSet.intersection(MapSet.new(schema_keys), MapSet.new(core_keys))

      assert MapSet.size(overlap) == 0,
             "Schema overlap detected: #{inspect(MapSet.to_list(overlap))}"
    end

    test "supported options include core generation keys" do
      supported = VLLM.supported_provider_options()
      core_keys = ReqLLM.Provider.Options.all_generation_keys()

      # All core keys should be supported (except meta-keys like :provider_options)
      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))
      missing = core_without_meta -- supported
      assert missing == [], "Missing core generation keys: #{inspect(missing)}"
    end

    test "provider_extended_generation_schema includes both base and provider options" do
      extended_schema = VLLM.provider_extended_generation_schema()
      extended_keys = extended_schema.schema |> Keyword.keys()

      # Should include all core generation keys
      core_keys = ReqLLM.Provider.Options.all_generation_keys()
      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))

      for core_key <- core_without_meta do
        assert core_key in extended_keys,
               "Extended schema missing core key: #{core_key}"
      end

      # Should include provider-specific keys
      provider_keys = VLLM.provider_schema().schema |> Keyword.keys()

      for provider_key <- provider_keys do
        assert provider_key in extended_keys,
               "Extended schema missing provider key: #{provider_key}"
      end
    end
  end

  describe "request preparation & pipeline wiring" do
    test "prepare_request creates configured request" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      prompt = "Hello world"
      opts = [temperature: 0.7, max_tokens: 100]

      {:ok, request} = VLLM.prepare_request(:chat, model, prompt, opts)

      assert %Req.Request{} = request
      assert request.url.path == "/chat/completions"
      assert request.method == :post
    end

    test "attach configures authentication and pipeline" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      opts = [temperature: 0.5, max_tokens: 50]

      request = Req.new() |> VLLM.attach(model, opts)

      # Verify authentication (optional for vLLM)
      _auth_header = Enum.find(request.headers, fn {name, _} -> name == "authorization" end)
      # Auth header may or may not be present depending on VLLM_API_KEY

      # Verify pipeline steps
      request_steps = Keyword.keys(request.request_steps)
      response_steps = Keyword.keys(request.response_steps)

      assert :llm_encode_body in request_steps
      assert :llm_decode_response in response_steps
    end

    test "error handling for invalid configurations" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      prompt = "Hello world"

      # Unsupported operation
      {:error, error} = VLLM.prepare_request(:unsupported, model, prompt, [])
      assert %ReqLLM.Error.Invalid.Parameter{} = error

      # Provider mismatch
      wrong_model = ReqLLM.Model.new(:openai, "gpt-4")

      assert_raise ReqLLM.Error.Invalid.Provider, fn ->
        Req.new() |> VLLM.attach(wrong_model, [])
      end
    end
  end

  describe "body encoding & context translation" do
    test "encode_body without tools" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      context = context_fixture()

      # Create a mock request with the expected structure
      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false
        ]
      }

      # Test the encode_body function directly
      updated_request = VLLM.encode_body(mock_request)

      assert is_binary(updated_request.body)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["model"] == "llama-3.1-8b-instant"
      assert is_list(decoded["messages"])
      assert length(decoded["messages"]) == 2
      assert decoded["stream"] == false
      refute Map.has_key?(decoded, "tools")

      [system_msg, user_msg] = decoded["messages"]
      assert system_msg["role"] == "system"
      assert user_msg["role"] == "user"
    end

    test "encode_body with vLLM-specific sampling options" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      context = context_fixture()

      test_cases = [
        # vLLM sampling parameters
        {[best_of: 3], fn json -> assert json["best_of"] == 3 end},
        {[use_beam_search: true], fn json -> assert json["use_beam_search"] == true end},
        {[min_p: 0.05], fn json -> assert json["min_p"] == 0.05 end},
        {[repetition_penalty: 1.2], fn json -> assert json["repetition_penalty"] == 1.2 end}
      ]

      for {provider_opts, assertion} <- test_cases do
        options = [context: context, model: model.model, stream: false] ++ provider_opts
        mock_request = %Req.Request{options: options}
        updated_request = VLLM.encode_body(mock_request)
        decoded = Jason.decode!(updated_request.body)
        assertion.(decoded)
      end
    end

    test "encode_body with vLLM guided decoding options" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      context = context_fixture()

      test_cases = [
        # Guided decoding options
        {[guided_json: %{"type" => "object", "properties" => %{"name" => %{"type" => "string"}}}],
         fn json ->
           assert json["guided_json"]["type"] == "object"
           assert json["guided_json"]["properties"]["name"]["type"] == "string"
         end},
        {[guided_regex: "^[A-Z][a-z]+$"], fn json -> assert json["guided_regex"] == "^[A-Z][a-z]+$" end},
        {[guided_choice: ["yes", "no", "maybe"]], fn json -> assert json["guided_choice"] == ["yes", "no", "maybe"] end},
        {[guided_grammar: "expr: term '+' term"], fn json -> assert json["guided_grammar"] == "expr: term '+' term" end}
      ]

      for {provider_opts, assertion} <- test_cases do
        options = [context: context, model: model.model, stream: false] ++ provider_opts
        mock_request = %Req.Request{options: options}
        updated_request = VLLM.encode_body(mock_request)
        decoded = Jason.decode!(updated_request.body)
        assertion.(decoded)
      end
    end

    test "encode_body handles standard OpenAI options" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      context = context_fixture()

      test_cases = [
        {[temperature: 0.2, max_tokens: 55, top_p: 0.9, frequency_penalty: 0.1],
         fn json ->
           assert json["temperature"] == 0.2
           assert json["max_tokens"] == 55
           assert json["top_p"] == 0.9
           assert json["frequency_penalty"] == 0.1
         end},
        {[presence_penalty: 0.2, user: "test_user", seed: 12345],
         fn json ->
           assert json["presence_penalty"] == 0.2
           assert json["user"] == "test_user"
           assert json["seed"] == 12345
         end}
      ]

      for {options, assertion} <- test_cases do
        full_options = [context: context, model: model.model, stream: false] ++ options
        mock_request = %Req.Request{options: full_options}
        updated_request = VLLM.encode_body(mock_request)
        decoded = Jason.decode!(updated_request.body)
        assertion.(decoded)
      end
    end
  end

  describe "response decoding & normalization" do
    test "decode_response handles non-streaming responses" do
      # Create a mock OpenAI-format response
      mock_json_response = openai_format_json_fixture()

      # Create a mock Req response
      mock_resp = %Req.Response{
        status: 200,
        body: mock_json_response
      }

      # Create a mock request with context
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, stream: false, model: "llama-3.1-8b-instruct"]
      }

      # Test decode_response directly
      {req, resp} = VLLM.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Response{} = resp.body

      response = resp.body
      assert is_binary(response.id)
      assert response.model == model.model
      assert response.stream? == false

      # Verify message normalization
      assert response.message.role == :assistant
      text = ReqLLM.Response.text(response)
      assert is_binary(text)
      assert String.length(text) > 0
      assert response.finish_reason in [:stop, :length, "stop", "length"]

      # Verify usage normalization
      assert is_integer(response.usage.input_tokens)
      assert is_integer(response.usage.output_tokens)
      assert is_integer(response.usage.total_tokens)

      # Verify context advancement (original + assistant)
      assert length(response.context.messages) == 3
      assert List.last(response.context.messages).role == :assistant
    end

    test "decode_response handles API errors with non-200 status" do
      # Create error response
      error_body = %{
        "error" => %{
          "message" => "Model not found",
          "type" => "not_found_error",
          "code" => "model_not_found"
        }
      }

      mock_resp = %Req.Response{
        status: 404,
        body: error_body
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, model: "llama-3.1-8b-instant"]
      }

      # Test decode_response error handling
      {req, error} = VLLM.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Error.API.Response{} = error
      assert error.status == 404
      assert error.response_body == error_body
    end
  end

  describe "object generation edge cases" do
    test "prepare_request for :object with default max_tokens" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      prompt = "Generate an object"
      {:ok, schema} = ReqLLM.Schema.compile([])

      # No max_tokens specified
      opts = [compiled_schema: schema]
      {:ok, request} = VLLM.prepare_request(:object, model, prompt, opts)

      # Should get default of 4096
      assert request.options[:max_tokens] == 4096
    end

    test "prepare_request rejects unsupported operations" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      prompt = "Hello world"

      # Test an unsupported operation
      {:error, error} = VLLM.prepare_request(:unsupported, model, prompt, [])
      assert %ReqLLM.Error.Invalid.Parameter{} = error
      assert error.parameter =~ "operation: :unsupported not supported"
    end
  end

  describe "usage extraction" do
    test "extract_usage with valid usage data" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")

      body_with_usage = %{
        "usage" => %{
          "prompt_tokens" => 10,
          "completion_tokens" => 20,
          "total_tokens" => 30
        }
      }

      {:ok, usage} = VLLM.extract_usage(body_with_usage, model)
      assert usage["prompt_tokens"] == 10
      assert usage["completion_tokens"] == 20
      assert usage["total_tokens"] == 30
    end

    test "extract_usage with missing usage data" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      body_without_usage = %{"choices" => []}

      {:error, :no_usage_found} = VLLM.extract_usage(body_without_usage, model)
    end

    test "extract_usage with invalid body type" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")

      {:error, :invalid_body} = VLLM.extract_usage("invalid", model)
      {:error, :invalid_body} = VLLM.extract_usage(nil, model)
      {:error, :invalid_body} = VLLM.extract_usage(123, model)
    end
  end

  describe "custom configuration" do
    test "custom base URL and API key override" do
      model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
      prompt = "Hello world"

      opts = [
        temperature: 0.7,
        provider_options: [
          base_url: "https://custom-vllm.example.com/v1",
          api_key: "custom-token-123"
        ]
      ]

      {:ok, request} = VLLM.prepare_request(:chat, model, prompt, opts)

      # Verify custom base URL is applied
      assert request.url.scheme == "https"
      assert request.url.host == "custom-vllm.example.com"
      assert request.url.port == 443

      # Verify custom API key is applied during encoding
      # The provider_options should already be in the request from prepare_request
      encoded_request = VLLM.encode_body(request)
      auth_headers = Map.get(encoded_request.headers, "authorization", [])
      assert "Bearer custom-token-123" in auth_headers
    end

    test "environment variable base URL override" do
      # Set environment variable
      original_env = System.get_env("VLLM_BASE_URL")
      System.put_env("VLLM_BASE_URL", "https://env-vllm.example.com/v1")

      try do
        model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
        prompt = "Hello world"
        opts = [temperature: 0.7]

        {:ok, request} = VLLM.prepare_request(:chat, model, prompt, opts)

        # Verify environment URL is applied
        assert request.url.scheme == "https"
        assert request.url.host == "env-vllm.example.com"
        assert request.url.port == 443
      after
        # Restore original environment
        if original_env do
          System.put_env("VLLM_BASE_URL", original_env)
        else
          System.delete_env("VLLM_BASE_URL")
        end
      end
    end

    test "provider_options base_url takes precedence over environment" do
      # Set environment variable
      original_env = System.get_env("VLLM_BASE_URL")
      System.put_env("VLLM_BASE_URL", "https://env-vllm.example.com/v1")

      try do
        model = ReqLLM.Model.new(:vllm, "llama-3.1-8b-instant")
        prompt = "Hello world"

        opts = [
          temperature: 0.7,
          provider_options: [
            base_url: "https://override-vllm.example.com/v1"
          ]
        ]

        {:ok, request} = VLLM.prepare_request(:chat, model, prompt, opts)

        # Verify provider_options takes precedence
        assert request.url.scheme == "https"
        assert request.url.host == "override-vllm.example.com"
        assert request.url.port == 443
      after
        # Restore original environment
        if original_env do
          System.put_env("VLLM_BASE_URL", original_env)
        else
          System.delete_env("VLLM_BASE_URL")
        end
      end
    end
  end
end