defmodule ReqLLM.Provider.OptionsTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Provider.Options

  # Mock provider for testing - implements minimal Provider behavior
  defmodule MockProvider do
    @behaviour ReqLLM.Provider

    def provider_id, do: :mock
    def default_base_url, do: "https://api.mock.com"
    def supported_provider_options, do: [:custom_option, :another_option]

    # Provider schema with custom options
    def provider_schema do
      NimbleOptions.new!(
        custom_option: [type: :string, doc: "Custom provider option"],
        another_option: [type: :integer, doc: "Another custom option"]
      )
    end

    # Translation function that renames max_tokens for o1 models
    def translate_options(:chat, %ReqLLM.Model{model: <<"o1", _::binary>>}, opts) do
      case Keyword.pop(opts, :max_tokens) do
        {nil, rest} ->
          {rest, []}

        {value, rest} ->
          new_opts = Keyword.put(rest, :max_completion_tokens, value)
          warning = "Renamed :max_tokens to :max_completion_tokens for o1 model"
          {new_opts, [warning]}
      end
    end

    def translate_options(_operation, _model, opts), do: {opts, []}

    # Required Provider callbacks (stubs)
    def attach(_request, _model, _opts), do: nil
    def prepare_request(_operation, _model, _input, _opts), do: {:error, :not_implemented}
    def encode_body(_request), do: nil
    def decode_response(_response), do: nil
  end

  # Provider without custom schema for fallback testing
  defmodule SimpleProvider do
    @behaviour ReqLLM.Provider

    def provider_id, do: :simple
    def default_base_url, do: "https://api.simple.com"
    def supported_provider_options, do: []

    # Required Provider callbacks (stubs)
    def attach(_request, _model, _opts), do: nil
    def prepare_request(_operation, _model, _input, _opts), do: {:error, :not_implemented}
    def encode_body(_request), do: nil
    def decode_response(_response), do: nil
  end

  describe "Options.process/4 - core functionality" do
    test "validates and passes through standard generation options" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}
      opts = [temperature: 0.7, max_tokens: 1000]

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)
      assert processed[:temperature] == 0.7
      assert processed[:max_tokens] == 1000
    end

    test "returns error for invalid generation options" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}
      opts = [temperature: "invalid"]

      assert {:error, %ReqLLM.Error.Unknown.Unknown{}} =
               Options.process(MockProvider, :chat, model, opts)
    end

    test "handles empty options with defaults" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}
      opts = []

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)
      # Should have default values
      assert processed[:stream] == false
      assert processed[:n] == 1
    end
  end

  describe "Options.process/4 - provider-specific options" do
    test "validates provider options nested under :provider_options key" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      opts = [
        temperature: 0.7,
        provider_options: [
          custom_option: "test_value",
          another_option: 42
        ]
      ]

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)
      assert processed[:temperature] == 0.7
      assert processed[:provider_options][:custom_option] == "test_value"
      assert processed[:provider_options][:another_option] == 42
    end

    test "rejects invalid provider options" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      opts = [
        temperature: 0.7,
        provider_options: [
          custom_option: "valid",
          # should be integer
          another_option: "invalid_type"
        ]
      ]

      assert {:error, %ReqLLM.Error.Unknown.Unknown{}} =
               Options.process(MockProvider, :chat, model, opts)
    end

    test "works with providers that have no custom schema" do
      model = %ReqLLM.Model{provider: :simple, model: "test-model"}
      opts = [temperature: 0.7]

      assert {:ok, processed} = Options.process(SimpleProvider, :chat, model, opts)
      assert processed[:temperature] == 0.7
      refute Keyword.has_key?(processed, :provider_options)
    end

    test "supports nested provider options" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      opts = [
        temperature: 0.8,
        provider_options: [
          custom_option: "nested_value",
          another_option: 200
        ]
      ]

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)
      assert processed[:temperature] == 0.8
      assert processed[:provider_options][:custom_option] == "nested_value"
      assert processed[:provider_options][:another_option] == 200
    end
  end

  describe "Options.process/4 - req_http_options handling" do
    test "preserves req_http_options for merging into Req request" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      opts = [
        temperature: 0.7,
        req_http_options: [
          timeout: 60_000,
          retry_attempts: 5
        ]
      ]

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)
      assert processed[:temperature] == 0.7
      assert processed[:req_http_options][:timeout] == 60_000
      assert processed[:req_http_options][:retry_attempts] == 5
    end

    test "handles missing req_http_options gracefully" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}
      opts = [temperature: 0.7]

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)
      assert processed[:temperature] == 0.7
      refute Keyword.has_key?(processed, :req_http_options)
    end
  end

  describe "Options.process/4 - provider translation" do
    test "applies provider-specific option translation" do
      model = %ReqLLM.Model{provider: :mock, model: "o1-preview"}
      opts = [temperature: 0.7, max_tokens: 1000]

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)

      # max_tokens should be renamed to max_completion_tokens
      assert processed[:temperature] == 0.7
      assert processed[:max_completion_tokens] == 1000
      refute Keyword.has_key?(processed, :max_tokens)
    end

    test "handles translation warnings based on on_unsupported setting" do
      import ExUnit.CaptureLog

      model = %ReqLLM.Model{provider: :mock, model: "o1-preview"}
      opts = [max_tokens: 1000, on_unsupported: :warn]

      log_output =
        capture_log(fn ->
          assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)
          assert processed[:max_completion_tokens] == 1000
        end)

      assert log_output =~ "Renamed :max_tokens to :max_completion_tokens"
    end

    test "translation works correctly for o1 models" do
      model = %ReqLLM.Model{provider: :mock, model: "o1-preview"}

      opts = [
        max_tokens: 1000,
        context: %ReqLLM.Context{messages: []}
      ]

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)

      # max_tokens should be translated to max_completion_tokens by the mock provider
      assert processed[:max_completion_tokens] == 1000
      refute Keyword.has_key?(processed, :max_tokens)
      assert %ReqLLM.Context{} = processed[:context]
    end
  end

  describe "Options.process/4 - internal keys handling" do
    test "preserves internal keys and bypasses validation" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      opts = [
        temperature: 0.7,
        # Internal keys that should bypass validation
        req_http_options: %{unknown: "value"},
        fixture: :test_fixture,
        operation: :embedding,
        text: "input text",
        context: %ReqLLM.Context{messages: []}
      ]

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)

      assert processed[:temperature] == 0.7
      assert processed[:req_http_options] == %{unknown: "value"}
      assert processed[:fixture] == :test_fixture
      assert processed[:operation] == :embedding
      assert processed[:text] == "input text"
      assert %ReqLLM.Context{} = processed[:context]
    end

    test "validates context when provided" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      # Valid context
      valid_opts = [temperature: 0.7, context: %ReqLLM.Context{messages: []}]
      assert {:ok, processed} = Options.process(MockProvider, :chat, model, valid_opts)
      assert %ReqLLM.Context{} = processed[:context]

      # Invalid context should raise specific error
      invalid_opts = [temperature: 0.7, context: "invalid"]

      assert_raise ReqLLM.Error.Invalid.Parameter, fn ->
        Options.process!(MockProvider, :chat, model, invalid_opts)
      end
    end
  end

  describe "Options.process/4 - error handling" do
    test "process/4 returns error tuples while process!/4 raises" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}
      invalid_opts = [temperature: "invalid"]

      # process/4 returns error tuple
      assert {:error, %ReqLLM.Error.Unknown.Unknown{}} =
               Options.process(MockProvider, :chat, model, invalid_opts)

      # process!/4 raises exception
      assert_raise NimbleOptions.ValidationError, fn ->
        Options.process!(MockProvider, :chat, model, invalid_opts)
      end
    end
  end

  describe "Options.process/4 - provider key collision detection" do
    # Provider that defines options conflicting with core generation options
    defmodule ConflictingProvider do
      @behaviour ReqLLM.Provider

      def provider_id, do: :conflicting
      def default_base_url, do: "https://api.conflicting.com"
      # These conflict!
      def supported_provider_options, do: [:temperature, :max_tokens]

      # Provider schema with conflicting options
      def provider_schema do
        NimbleOptions.new!(
          temperature: [type: :string, doc: "Conflicting temperature option"],
          max_tokens: [type: :string, doc: "Conflicting max_tokens option"],
          safe_option: [type: :integer, doc: "Non-conflicting option"]
        )
      end

      # Required Provider callbacks (stubs)
      def attach(_request, _model, _opts), do: nil
      def prepare_request(_operation, _model, _input, _opts), do: {:error, :not_implemented}
      def encode_body(_request), do: nil
      def decode_response(_response), do: nil
    end

    test "detects when provider options shadow core generation options" do
      model = %ReqLLM.Model{provider: :conflicting, model: "test-model"}
      opts = [temperature: 0.7]

      assert_raise ReqLLM.Error.Invalid.Parameter, fn ->
        Options.process!(ConflictingProvider, :chat, model, opts)
      end
    end

    test "provides helpful error message about conflicting keys" do
      model = %ReqLLM.Model{provider: :conflicting, model: "test-model"}
      opts = [temperature: 0.7]

      error =
        assert_raise ReqLLM.Error.Invalid.Parameter, fn ->
          Options.process!(ConflictingProvider, :chat, model, opts)
        end

      assert error.parameter =~
               "Provider conflicting defines options that shadow core generation options"

      assert error.parameter =~ "max_tokens, temperature"
      assert error.parameter =~ "Provider-specific options must not conflict"
    end

    test "allows providers with non-conflicting options to work normally" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      opts = [
        temperature: 0.7,
        provider_options: [custom_option: "test", another_option: 42]
      ]

      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts)
      assert processed[:temperature] == 0.7
      assert processed[:provider_options][:custom_option] == "test"
    end
  end

  describe "Options.process/4 - enhanced error messages" do
    test "suggests provider_options for unknown options that match provider schema" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      opts = [
        temperature: 0.7,
        # This matches provider schema
        custom_option: "should be nested",
        unknown_option: "totally unknown"
      ]

      case Options.process(MockProvider, :chat, model, opts) do
        {:error,
         %ReqLLM.Error.Unknown.Unknown{error: %NimbleOptions.ValidationError{message: message}}} ->
          assert message =~ "Suggestion: The following options appear to be provider-specific"
          assert message =~ "custom_option"
          assert message =~ "provider_options"

        other ->
          flunk("Expected enhanced validation error, got: #{inspect(other)}")
      end
    end

    test "provides helpful suggestions for similar option names" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      opts = [
        temperature: 0.7,
        # Similar to custom_option
        custon_option: "typo in custom_option"
      ]

      case Options.process(MockProvider, :chat, model, opts) do
        {:error,
         %ReqLLM.Error.Unknown.Unknown{error: %NimbleOptions.ValidationError{message: message}}} ->
          assert message =~ "Did you mean one of these provider-specific options"
          assert message =~ "custon_option -> custom_option"

        other ->
          flunk("Expected enhanced validation error, got: #{inspect(other)}")
      end
    end

    test "provides tips for invalid value errors" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}
      # Should be a number
      opts = [temperature: "invalid"]

      case Options.process(MockProvider, :chat, model, opts) do
        {:error,
         %ReqLLM.Error.Unknown.Unknown{error: %NimbleOptions.ValidationError{message: message}}} ->
          assert message =~ "Tip: Check the documentation for valid parameter ranges"
          assert message =~ "provider_options key"

        other ->
          flunk("Expected enhanced validation error, got: #{inspect(other)}")
      end
    end
  end

  describe "Options.process/4 - edge cases" do
    test "handles stream/stream? alias conversion" do
      model = %ReqLLM.Model{provider: :mock, model: "test-model"}

      # stream? should be converted to stream
      opts_with_alias = [stream?: true, temperature: 0.7]
      assert {:ok, processed} = Options.process(MockProvider, :chat, model, opts_with_alias)
      assert processed[:stream] == true
      refute Keyword.has_key?(processed, :stream?)
    end

    test "handles provider without translate_options callback" do
      model = %ReqLLM.Model{provider: :simple, model: "test-model"}
      opts = [temperature: 0.7, max_tokens: 1000]

      assert {:ok, processed} = Options.process(SimpleProvider, :chat, model, opts)
      assert processed[:temperature] == 0.7
      assert processed[:max_tokens] == 1000
    end
  end
end
