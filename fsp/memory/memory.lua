local class = require 'class'
local memory = class('ObsMemory')


function memory:__init(params)
  if not params.capacity or not params.stateDim then
    error("capacity and stateDim params required by transition memory")
  end
  self._capacity = params.capacity
  self.stateDim = params.stateDim
  self.o = torch.ByteTensor(self._capacity, self.stateDim):zero()
  self:reset()
end

function memory:reset()
  self._length = 0
  self._added = 0
  self._head = 0
  self._tail = 1
end

function memory:length()
  return self._length
end

function memory:capacity()
  return self._capacity
end

function memory:full()
  return self._length >= self._capacity
end

function memory:add(data)
  self._head = self._head%self._capacity+1
  self:set(self._head, data)
  if self._length < self._capacity then
    -- ToDo: careful (with cursor) in case resizing becomes supported
    self._length = self._length + 1
  else
    self._tail = self._tail%self._capacity+1
  end
end

function memory:add_batch(nBatch, data)
  self._head = (self._head + nBatch - 1)%self._capacity + 1
  local batchTail = self._head - nBatch + 1
  if batchTail < 1 then
    self:set_batch(data, self._head, 1, 1)
    local bufTail = self._capacity - nBatch + self._head + 1
    self:set_batch(data, nBatch-self._head, bufTail, self._head+1)
  else
    self:set_batch(data, nBatch, batchTail)
  end
  self._length = self._length + nBatch
  if self._length > self._capacity then
    self._tail = (self._tail + self._length - 1)%self._capacity + 1
    self._length = self._capacity
  end
  assert((self._head - self._tail)%self._capacity + 1 == self._length, self._head.." - "..self._tail.." + 1 != "..self._length)
end

function memory:sample()
  return self:get(torch.random(self._length))
end


local reservoir = class('ObsReservoirMemory', 'ObsMemory')

function reservoir:__init(params)
  memory.__init(self, params)
  self.exp_beta = params.exp_beta or 0
end

function reservoir:add(data)
  if self._length < self._capacity then
    self._length = self._length + 1
    self:set(self._length, data)
  else
    if self.exp_beta > 0 then
      -- exponential reservoir sampling
      if torch.FloatTensor.torch.uniform() < self._capacity/math.min(self.exp_beta, self._added+1) then
        self:set(torch.random(self._capacity), data)
      end
    else
      -- vanilla reservoir sampling
      if torch.random(self._added+1) <= self._capacity then
        self:set(torch.random(self._capacity), data)
      end
    end
  end
  self._added = self._added + 1
end


return {ObsMemory = memory, ObsReservoirMemory = reservoir}
