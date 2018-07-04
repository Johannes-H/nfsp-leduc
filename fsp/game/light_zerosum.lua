local class = require 'class'

local game = class('LightZeroSumGame')

function game:__init(sim, obs)
  self.simulator = sim
  if not obs then
    error("Require obs param")
  end
  self.nPlayers = 2
  self.state = self.simulator:new_state(obs)
  self.r_t = torch.Tensor(self.nPlayers)
  self.c_t = torch.Tensor(self.nPlayers)
  self.lastRewards = self.r_t:clone()
  self:reset()
end

function game:reset()
  self.state:reset()
  self.r_t:zero()
  self.c_t:fill(1)
end

function game:player()
  return self.state:player()
end

function game:act(action)
  self.simulator:step(self.state, action, self.lastRewards)
  self.r_t:addcmul(self.lastRewards, self.c_t)
  if self.state.terminal then
    self.c_t:mul(0)
    self.state:reset()
  end

  local player = self:player()
  local o_t = self.state:update_observation()
  local r_t = self.r_t[player]
  local c_t = self.c_t[player]
  self.c_t[player] = 1
  self.r_t[player] = 0
  return o_t, r_t, c_t
end

return {LightZeroSum = game}
