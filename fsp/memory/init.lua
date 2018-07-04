local tablex = require 'pl.tablex'

local memories = {}

tablex.update(memories, require 'fsp.memory.transition')
tablex.update(memories, require 'fsp.memory.actions')
--
function memories.createTransitionBuffer(mbSize, stateDim, gpu)
  local buffer
  if gpu then
    buffer = {o_tm1 = torch.CudaTensor(mbSize, stateDim), a_tm1 = torch.LongTensor(mbSize), r_t = torch.FloatTensor(mbSize), o_t = torch.CudaTensor(mbSize, stateDim), c_t = torch.FloatTensor(mbSize)}
  else
    buffer = {o_tm1 = torch.FloatTensor(mbSize, stateDim), a_tm1 = torch.LongTensor(mbSize), r_t = torch.FloatTensor(mbSize), o_t = torch.FloatTensor(mbSize, stateDim), c_t = torch.FloatTensor(mbSize)}
  end
  return buffer
end

function memories.createActionBuffer(mbSize, stateDim, gpu)
  local buffer
  if gpu then
    buffer = {obs = torch.CudaTensor(mbSize, stateDim), actions = torch.LongTensor(mbSize)}
  else
    buffer = {obs = torch.FloatTensor(mbSize, stateDim), actions = torch.LongTensor(mbSize)}
  end
  return buffer
end

return memories
