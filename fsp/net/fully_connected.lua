require "nn"
require "nngraph"

local function full1(nIn, nOut, HUs, softmax)
  local nonLinearity = nn.ReLU

  local input = nn.Identity()()
  local inputLayer = input
  local nLayers = #HUs
  local layers = {}
  layers[0] = inputLayer
  HUs[0] = nIn
  for i=1,nLayers do
    layers[i] = nonLinearity()(nn.Linear(HUs[i-1], HUs[i])(layers[i-1]))
  end
  local lastLayer = layers[nLayers]
  local output = nn.Linear(HUs[nLayers],nOut)(lastLayer)
  if softmax then
    output = nn.SoftMax()(output)
  end
  local net = nn.gModule({input}, {output})
  local params, _ = net:getParameters()
  print("Created model 1 with "..params:size(1).." parameters")
  return net
end

return {Full1 = full1}
