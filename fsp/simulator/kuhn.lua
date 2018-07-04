local torch = require 'torch'
local class = require 'class'
local state = class('KuhnState')

local CHECKFOLD = 1
local BETCALL = 2

local REW_SCALE = 0.33

local function sampleCard(deadCards)
  local card = 0
  repeat
    card = math.random(3)
  until deadCards[card]==0
  deadCards[card] = 1
  return card
end

function state:__init(obs)
  self.nPlayers = 2
  self.obs = obs or torch.Tensor(6)
  self.playerBet = torch.DoubleTensor(self.nPlayers)
  self.playerFolded = torch.ByteTensor(self.nPlayers)
  self.deadCards = torch.ByteTensor(3)
  self.playerHoldings = torch.ByteTensor(self.nPlayers)
  self:reset()
end

function state:reset()
  self.playerBet:fill(1)
  self.playerFolded:zero()
  self.deadCards:zero()
  self.playerHoldings[1] = sampleCard(self.deadCards)
  self.playerHoldings[2] = sampleCard(self.deadCards)
  self.currentBet = 0
  self.potsize = 2
  self.playersTurn = 1
  self.terminal = false
  self:update_observation()
end

function state:num_players()
    return self.nPlayers
end

function state:player()
  return self.playersTurn
end

function state:observe()
  return self.obs
end

function state:update_observation()
  self.obs:zero()
  self.obs[self.playerHoldings[self.playersTurn]] = 1
  if self.playersTurn == 2 then
    if self.playerBet[1] > 1.5 then
      self.obs[5] = 1 -- p1 bet
    else
      self.obs[4] = 1 -- p1 check
    end
  else
    if self.playerBet[2] > 1.5 then
      self.obs[4] = 1 -- p1 check
      self.obs[6] = 1  -- p2 bet
    end

  end
  return self.obs
end


local simulator = class('KuhnSimulator')

function simulator:__init(args)
end

function simulator:actionDim()
  return 2
end

function simulator:stateDim()
  return self:new_state():observe():size(1)
end


function simulator:new_state(obs)
  return state(obs)
end

function simulator:step(state, action, rewards)
  rewards:zero()
  if action == CHECKFOLD then
    if state.currentBet == 0 then
      -- check
      local pt = state.playersTurn + 1
      state.playersTurn = pt
      if pt > state.nPlayers then
        state.terminal = true
      end
    else
      -- fold
      -- assuming 2-player Kuhn
      state.playerFolded[state.playersTurn] = 1
      state.terminal = true
    end
  elseif action == BETCALL then
    local pt = state.playersTurn
    rewards[pt] = -1
    state.playerBet[pt] = state.playerBet[pt] + 1
    state.potsize = state.potsize + 1
    state.playersTurn = pt%state.nPlayers+1
    if state.currentBet == 0 then
      state.currentBet = state.currentBet + 1
    else
      state.terminal = true
    end
  else
    error("invalid kuhn action")
  end
  if state.terminal then
    -- compute terminal rewards
    simulator:compute_terminal_rewards(state, rewards)
  end
  rewards:mul(REW_SCALE)
  return action
end

function simulator:compute_terminal_rewards(state, rewards)
  -- assuming 2-player kuhn
  winningPlayer = -1
  if state.playerFolded[1] == 1 then
    winningPlayer = 2
  elseif state.playerFolded[2] == 1 then
    winningPlayer = 1
  else
    if state.playerHoldings[1] > state.playerHoldings[2] then
      winningPlayer = 1
    else
      winningPlayer = 2
    end
  end
  rewards[winningPlayer] = rewards[winningPlayer] + state.potsize -- using relative rewards, thus whole pot is won
end


return {Kuhn = simulator}
