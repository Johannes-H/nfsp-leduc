local class = require 'class'
local games = require 'fsp.game'
-- local agents = require 'fsp.agent'

local gen = class('ZeroSumDataGenerator')


function gen:__init(params)
  self.nPlayers = 2
  self.nGames = params.nGames
  self.agents = params.agents
  self.agents_forward = params.agents_forward
  self._stateDim = params.stateDim
  if #self.agents[1] ~= self.nGames then
    error("No of agents does not match nGames")
  end
  self.batch_o_t = torch.Tensor(self.nGames, self._stateDim)
  self.batch_r_t = torch.Tensor(self.nGames):zero()
  self.batch_c_t = torch.ByteTensor(self.nGames):zero()
  self.games = {}
  for i=1,self.nGames do
    self.games[i] = games.LightZeroSum(params.simulator, self.batch_o_t[i])
  end
end

function gen:setAnticipatory(anticipatory, player)
  if player then
    for i=1,self.nGames do
      self.agents[player][i]:setAnticipatory(anticipatory)
    end
  else
    for p=1,2 do
      for i=1,self.nGames do
        self.agents[p][i]:setAnticipatory(anticipatory)
      end
    end
  end
end

function gen:setExploration(explo, player)
  if player then
    for i=1,self.nGames do
      self.agents[player][i]:setExploration(explo)
    end
  else
    for p=1,2 do
      for i=1,self.nGames do
        self.agents[p][i]:setExploration(explo)
      end
    end
  end
end

function gen:stateDim()
  return self._stateDim
end

function gen:generate()
  -- run all agents/nets forward (e.g. br/avg for both players)
  self.agents_forward(self.batch_o_t)
  -- loop through games/agents and simulate actions
  local player, action
  for i=1,self.nGames do
    player = self.games[i]:player()
    action = self.agents[player][i]:act(self.batch_o_t[i], self.batch_r_t[i], self.batch_c_t[i])
     _, self.batch_r_t[i], self.batch_c_t[i] = self.games[i]:act(action)
  end
end

return {ZeroSumGenerator = gen}
