local class = require 'class'

local agent = class('FSPAgent')

function agent:__init(params)
  self.br = params.br
  self.avg = params.avg
  self.current_policy = self.avg
  self.track_behaviour = false
  if not params.anticipatory then error("Anticipatory param reqired") end
  self:setAnticipatory(params.anticipatory)
  self.mem_rl = params.mem_rl
  self.mem_sl = params.mem_sl
  self.o_tm1 = nil
  self.a_tm1 = nil
end

function agent:setAnticipatory(anticipatory)
  self.anticipatory = anticipatory
end

function agent:setExploration(explo)
  self.br:setExploration(explo)
end

function agent:act(o_t, r_t, c_t)
  if c_t == 0 then
    self:sample_policy()
  end
  local a_t = self.current_policy:action()
  if self.track_behaviour then
    self.mem_sl:add{o_tm1 = o_t, a_tm1 = a_t}
  end
  if self.o_tm1 then
    self.mem_rl:add{o_tm1 = self.o_tm1, a_tm1 = self.a_tm1, r_t = r_t, o_t = o_t, c_t = c_t}  -- storing full transitions
    self.o_tm1:copy(o_t)
    self.a_tm1 = a_t
  else
    self.o_tm1 = o_t:clone()
    self.a_tm1 = a_t
  end
  return a_t
end

function agent:sample_policy()
  if torch.FloatTensor.torch.uniform() < self.anticipatory then
    self.current_policy = self.br
    self.track_behaviour = true
  else
    self.current_policy = self.avg
    self.track_behaviour = false
  end
end


local function createLightAgents(params)
  local policy_no_from = params.policy_no_from or 1
  local policy_no_to = policy_no_from+params.nAgents-1
  local agents = {}
  for i=policy_no_from, policy_no_to do
    agents[i] = agent{br = params.light_br:light_clone(i), avg = params.light_avg:light_clone(i), anticipatory = params.anticipatory, mem_rl = params.mem_rl, mem_sl = params.mem_sl}
  end
  return agents
end

return {FSP = agent, CreateMultipleFSP = createLightAgents}
