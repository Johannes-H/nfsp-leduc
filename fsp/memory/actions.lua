local class = require 'class'
local base = require 'fsp.memory.memory'

local reservoir = class('ActionReservoirMemory', 'ObsReservoirMemory')

function reservoir:__init(params)
  base.ObsReservoirMemory.__init(self, params)
  self.a = torch.LongTensor(self._capacity):zero()
  print("Setup ActionReservoirMemory with capacity/exp_beta: "..self._capacity/self.exp_beta)
end

function reservoir:set(idx, data)
  self.o[idx]:copy(data.o_tm1)
  self.a[idx] = data.a_tm1
end

function reservoir:get(idx)
  if idx > self._length then
    error("invalid idx (>length) in memory:get")
  end
  return self.o[idx], self.a[idx]
end

function reservoir:fill(buffer)
  local o_tm1, a_tm1
  local buf_o_tm1, buf_a_tm1 = buffer.obs, buffer.actions
  local bufDim = buf_o_tm1:size(1)
  for i=1,bufDim do
    o_tm1, a_tm1 = self:sample()
    buf_o_tm1[i]:copy(o_tm1)
    buf_a_tm1[i] = a_tm1
  end
end


return {ActionReservoir = reservoir}
