local class = require 'class'
local base = require 'fsp.memory.memory'

local memory = class('TransitionMemory', 'ObsMemory')

function memory:__init(params)
  base.ObsMemory.__init(self, params)
  self.a = torch.LongTensor(self._capacity):zero()
  self.r = torch.FloatTensor(self._capacity):zero()
  self.t = torch.ByteTensor(self._capacity):zero()
end

function memory:add(episode)
  for i=1,episode.length do
    self:_add(episode[i])
  end
end

function memory:_add(transition)
  self.o[self.cursor]:copy(transition.o_tm1)
  self.a[self.cursor] = transition.a_tm1
  self.r[self.cursor] = transition.r_t
  self.t[self.cursor] = transition.t_t
  self.cursor = self.cursor%self._capacity+1
  if self._length < self._capacity then
    -- ToDo: careful (with cursor) in case resizing becomes supported
    self._length = self._length + 1
  end
end

function memory:sample()
  local idx
  repeat
    idx = torch.random(1, self._length)  -- max length allowed too in case it is == capacity
  until idx ~= self.cursor-1
  return self:get(idx)
end

function memory:get(idx)
  if idx == self.cursor-1 then  -- == length if below capacity
    -- wrap-over of ringbuff at this position; thus transitions dont fit together here
    error("invalid idx (at cursor position, ringbuff seam) in memory:get")
  end
  if idx > self._length then
    error("invalid idx (>length) in memory:get")
  end
  local next_idx = idx%self._capacity+1  -- first check against self.cursor-1 covers length problems
  return self.o[idx], self.a[idx], self.r[idx], self.o[next_idx], self.t[idx]
end

function memory:fill_buffer(buf_o_tm1, buf_a_tm1, buf_r_t, buf_o_t, buf_t_t)
  local bufDim = buf_o_tm1:size(1)
  local o_tm1, a_tm1, r_t, o_t, t_t
  for i=1,bufDim do
    o_tm1, a_tm1, r_t, o_t, t_t = self:sample()
    buf_o_tm1[i]:copy(o_tm1)
    buf_a_tm1[i] = a_tm1
    buf_r_t[i] = r_t
    buf_o_t[i]:copy(o_t)
    buf_t_t[i] = t_t
  end
end


local whole = class('WholeTransitionMemory', 'ObsMemory')

function whole:__init(params)
  base.ObsMemory.__init(self, params)
  self.a = torch.LongTensor(self._capacity):zero()
  self.r = torch.FloatTensor(self._capacity):zero()
  self.o_ = torch.ByteTensor(self._capacity, self.stateDim):zero()
  self.c = torch.ByteTensor(self._capacity):zero()
end

function whole:set(idx, data)
  self.o[idx]:copy(data.o_tm1)
  self.o_[idx]:copy(data.o_t)
  self.a[idx] = data.a_tm1
  self.r[idx] = data.r_t
  self.c[idx] = data.c_t
end

function whole:get(idx)
  if idx > self._length then
    error("invalid idx (>length) in memory:get")
  end
  return self.o[idx], self.a[idx], self.r[idx], self.o_[idx], self.c[idx]
end

function whole:fill(buffer)
  local o_tm1, a_tm1, r_t, o_t, c_t
  local buf_o_tm1, buf_a_tm1, buf_r_t, buf_o_t, buf_c_t = buffer.o_tm1, buffer.a_tm1, buffer.r_t, buffer.o_t, buffer.c_t
  local bufDim = buf_o_tm1:size(1)
  for i=1,bufDim do
    o_tm1, a_tm1, r_t, o_t, c_t = self:sample()
    buf_o_tm1[i]:copy(o_tm1)
    buf_a_tm1[i] = a_tm1
    buf_r_t[i] = r_t
    buf_o_t[i]:copy(o_t)
    buf_c_t[i] = c_t
  end
end

function whole:fill_sl(buffer)
  local o_tm1, a_tm1
  local buf_o_tm1, buf_a_tm1 = buffer.obs, buffer.actions
  local bufDim = buf_o_tm1:size(1)
  for i=1,bufDim do
    o_tm1, a_tm1 = self:sample()
    buf_o_tm1[i]:copy(o_tm1)
    buf_a_tm1[i] = a_tm1
  end
end


return {Transition = memory, WholeTransition = whole}
