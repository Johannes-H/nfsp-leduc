local util = {}

local TOLERANCE = 1e-6

function util.sample(probs)
  local unirand = torch.FloatTensor.torch.uniform()
  for i=1,probs:size(1) do
    unirand = unirand - probs[i]
    if unirand < 0 then
      return i
    end
  end
  if unirand > TOLERANCE then
    error("fsp.sample ran through with unirand left: "..unirand)
  else
    return probs:size(1)
  end
end

function util.sampleActions(probs)
  local nActions = probs:size(1)
  local actions = torch.LongTensor(nActions)
  for i=1,nActions do
    actions[i] = util.sample(probs[i])
  end
  return actions
end

function util.linearDecay(base, const, counter)
  return base / (1+const*counter)
end

function util.polynomialDecay(base, const, exponent, counter)
  return base / (1+const*math.pow(counter, exponent))
end

return util
