defmodule ReqLLM.Step.Usage do
  @moduledoc """
  Centralized Req step that extracts token usage from provider responses,
  normalizes usage values across providers, computes costs, and emits telemetry.

  This step:
  * Extracts token usage numbers from provider responses
  * Normalizes usage data across different provider formats
  * Calculates costs using ReqLLM.Model cost metadata
  * Stores usage data in `response.private[:req_llm][:usage]`
  * Emits telemetry events for monitoring

  ## Usage

      request
      |> ReqLLM.Step.Usage.attach(model)

  ## Telemetry Events

  Emits `[:req_llm, :token_usage]` events with:
  * Measurements: `%{tokens: %{input: 123, output: 456, reasoning: 64}, cost: 0.0123}`
  * Metadata: `%{model: %ReqLLM.Model{}}`
  """

  @event [:req_llm, :token_usage]

  @doc """
  Attaches the Usage step to a Req request.

  ## Parameters

  - `req` - The Req.Request struct
  - `model` - Optional ReqLLM.Model struct for cost calculation

  ## Examples

      request
      |> ReqLLM.Step.Usage.attach(model)

  """
  @spec attach(Req.Request.t(), ReqLLM.Model.t() | nil) :: Req.Request.t()
  def attach(%Req.Request{} = req, model \\ nil) do
    req
    |> Req.Request.append_response_steps(llm_usage: &__MODULE__.handle/1)
    |> then(fn r ->
      if model, do: Req.Request.put_private(r, :req_llm_model, model), else: r
    end)
  end

  @doc false
  @spec handle({Req.Request.t(), Req.Response.t()}) :: {Req.Request.t(), Req.Response.t()}
  def handle({req, resp}) do
    provider_module = get_provider_module(req)

    with {:ok, usage} <- extract_usage(resp.body, provider_module),
         {:ok, model} <- fetch_model(req),
         {:ok, cost_breakdown} <- compute_cost_breakdown(usage, model) do
      # Keep legacy total cost for telemetry compatibility
      total_cost = cost_breakdown && cost_breakdown.total_cost
      meta = %{tokens: usage, cost: total_cost}

      # Add cost breakdown to meta if available
      meta =
        if cost_breakdown do
          Map.merge(meta, %{
            input_cost: cost_breakdown.input_cost,
            output_cost: cost_breakdown.output_cost,
            total_cost: cost_breakdown.total_cost
          })
        else
          meta
        end

      # Emit telemetry event for monitoring
      :telemetry.execute(@event, meta, %{model: model})

      # Store usage data in response private for access by callers
      req_llm_data = Map.get(resp.private, :req_llm, %{})
      updated_req_llm_data = Map.put(req_llm_data, :usage, meta)

      # Update Response.usage field with cost information if resp.body is a Response
      updated_resp =
        case resp.body do
          %ReqLLM.Response{usage: response_usage}
          when is_map(response_usage) and cost_breakdown != nil ->
            cached_tokens = usage[:cached_input] || 0

            augmented_usage =
              response_usage
              |> Map.put_new(:input_tokens, usage.input)
              |> Map.put_new(:output_tokens, usage.output)
              |> Map.put_new(:total_tokens, usage.input + usage.output)
              |> then(fn m ->
                if cached_tokens > 0, do: Map.put(m, :cached_tokens, cached_tokens), else: m
              end)
              |> Map.merge(%{
                input_cost: cost_breakdown.input_cost,
                output_cost: cost_breakdown.output_cost,
                total_cost: cost_breakdown.total_cost
              })

            updated_body = %{resp.body | usage: augmented_usage}
            %{resp | body: updated_body}

          _ ->
            resp
        end

      updated_resp = %{
        updated_resp
        | private: Map.put(updated_resp.private, :req_llm, updated_req_llm_data)
      }

      {req, updated_resp}
    else
      _ -> {req, resp}
    end
  end

  @spec get_provider_module(Req.Request.t()) :: module() | nil
  defp get_provider_module(%Req.Request{options: options}) do
    case options[:model] do
      %ReqLLM.Model{provider: provider_id} ->
        case ReqLLM.Provider.Registry.get_provider(provider_id) do
          {:ok, module} -> module
          _ -> nil
        end

      _ ->
        nil
    end
  end

  @spec extract_usage(any, module() | nil) :: {:ok, map()} | :error
  defp extract_usage(body, provider_module) do
    case provider_module do
      nil -> fallback_extract_usage(body)
      module -> provider_extract_usage(body, module) || fallback_extract_usage(body)
    end
  end

  defp provider_extract_usage(body, module) when is_atom(module) do
    if function_exported?(module, :extract_usage, 2) do
      case module.extract_usage(body, nil) do
        {:ok, usage} -> {:ok, normalize_usage(usage)}
        _ -> nil
      end
    end
  end

  @spec fallback_extract_usage(any) :: {:ok, map()} | :error
  defp fallback_extract_usage(%{"usage" => usage}) when is_map(usage) do
    {:ok, normalize_usage(usage)}
  end

  defp fallback_extract_usage(%{"prompt_tokens" => input, "completion_tokens" => output}) do
    {:ok, %{input: input, output: output, reasoning: 0, cached_input: 0}}
  end

  defp fallback_extract_usage(%{"input_tokens" => input, "output_tokens" => output}) do
    {:ok, %{input: input, output: output, reasoning: 0, cached_input: 0}}
  end

  defp fallback_extract_usage(%ReqLLM.Response{usage: usage}) when is_map(usage) do
    {:ok, normalize_usage(usage)}
  end

  defp fallback_extract_usage(_), do: :error

  @spec normalize_usage(map()) :: map()
  defp normalize_usage(usage) when is_map(usage) do
    %{
      input:
        usage[:input] || usage["input"] || usage["prompt_tokens"] || usage[:prompt_tokens] ||
          usage["input_tokens"] || usage[:input_tokens] || 0,
      output:
        usage[:output] || usage["output"] || usage["completion_tokens"] ||
          usage[:completion_tokens] || usage["output_tokens"] || usage[:output_tokens] || 0,
      reasoning: usage[:reasoning] || usage["reasoning"] || get_reasoning_tokens(usage) || 0,
      cached_input: get_cached_input_tokens(usage)
    }
  end

  defp get_reasoning_tokens(usage) do
    reasoning =
      get_in(usage, ["completion_tokens_details", "reasoning_tokens"]) ||
        get_in(usage, [:completion_tokens_details, :reasoning_tokens])

    case reasoning do
      n when is_integer(n) -> n
      _ -> 0
    end
  end

  defp get_cached_input_tokens(usage) do
    cached =
      usage[:cached_input] || usage["cached_input"] ||
        usage[:cached_tokens] || usage["cached_tokens"] ||
        get_in(usage, ["prompt_tokens_details", "cached_tokens"]) ||
        get_in(usage, [:prompt_tokens_details, :cached_tokens])

    input_tokens =
      usage[:input] || usage["input"] || usage["prompt_tokens"] || usage[:prompt_tokens] ||
        usage["input_tokens"] || usage[:input_tokens] || 0

    clamp_tokens(cached, input_tokens)
  end

  @spec fetch_model(Req.Request.t()) :: {:ok, ReqLLM.Model.t()} | :error
  defp fetch_model(%Req.Request{private: private, options: options}) do
    case private[:req_llm_model] || options[:model] do
      %ReqLLM.Model{} = model -> {:ok, model}
      _ -> :error
    end
  end

  @spec compute_cost_breakdown(map(), ReqLLM.Model.t()) ::
          {:ok, %{input_cost: float(), output_cost: float(), total_cost: float()} | nil}
  defp compute_cost_breakdown(%{input: _input_tokens, output: _output_tokens}, %ReqLLM.Model{
         cost: nil
       }) do
    {:ok, nil}
  end

  defp compute_cost_breakdown(
         %{input: input_tokens, output: output_tokens} = usage,
         %ReqLLM.Model{cost: cost_map}
       )
       when is_map(cost_map) do
    input_rate = cost_map[:input] || cost_map["input"]
    output_rate = cost_map[:output] || cost_map["output"]

    cached_rate =
      cost_map[:cached_input] || cost_map["cached_input"] ||
        cost_map[:cache_read] || cost_map["cache_read"] ||
        input_rate

    with {:ok, input_num} <- safe_to_number(input_tokens),
         {:ok, output_num} <- safe_to_number(output_tokens),
         true <- input_rate != nil and output_rate != nil do
      # Extract cached tokens and calculate split
      cached_tokens = clamp_tokens(Map.get(usage, :cached_input, 0), input_num)

      uncached_tokens = max(input_num - cached_tokens, 0)

      # Calculate costs with cached vs uncached rates
      input_cost =
        Float.round(
          uncached_tokens / 1000 * input_rate + cached_tokens / 1000 * cached_rate,
          6
        )

      output_cost = Float.round(output_num / 1000 * output_rate, 6)
      total_cost = Float.round(input_cost + output_cost, 6)

      {:ok,
       %{
         input_cost: input_cost,
         output_cost: output_cost,
         total_cost: total_cost
       }}
    else
      _ -> {:ok, nil}
    end
  end

  # Safely clamps a value to a valid token count within bounds.
  # Converts the value to a number and clamps it between 0 and the maximum allowed value.
  # Returns 0 if the value cannot be converted to a number.
  @spec clamp_tokens(any(), number()) :: integer()
  defp clamp_tokens(value, max_allowed) do
    case safe_to_number(value) do
      {:ok, int} ->
        int
        |> max(0)
        |> min(max(max_allowed, 0))

      _ ->
        0
    end
  end

  @spec safe_to_number(any()) :: {:ok, number()} | :error
  defp safe_to_number(value) when is_integer(value), do: {:ok, value}
  defp safe_to_number(value) when is_float(value), do: {:ok, trunc(value)}
  defp safe_to_number(_), do: :error
end
