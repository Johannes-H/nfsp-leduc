local torch = require 'torch'
local class = require 'class'
local histories = require 'fsp.simulator.history'

local PREFLOP = 1
local FLOP = 2

-- base actions
local FOLD = 1
local CHECKCALL = 2
local BETRAISE = 3
-- history actions
local CHECK = 2
local CALL = 3
local BET = 4
local RAISE = 5

local ANTE = 1
local SMALL_BET = 2
local BIG_BET = 4

local REW_SCALE = 0.0385

local function sampleCard(deadCards)
  local card = 0
  repeat
    card = math.random(6)
  until deadCards[card]==0
  deadCards[card] = 1
  return card%3+1 -- return rank instead of card, as suit irrelevant
end

local seqstate = class('LeducSeqState')


function seqstate:__init(obs)
  self.nPlayers = 2
  self.history = histories.HULevel{numRounds = 2, numRaises = 2}
  local len_cards = 6
  local len_hist = self.history:observe():size(1)
  self.obs = obs or torch.Tensor(len_cards + len_hist)
  self.obs_hist = self.obs:sub(len_cards+1,len_cards + len_hist)  -- into this sub tensor history will be copied in
  self.playerBet = torch.FloatTensor(self.nPlayers)
  self.playerFolded = torch.ByteTensor(self.nPlayers)
  self.deadCards = torch.ByteTensor(len_cards)
  self.playerHoldings = torch.ByteTensor(self.nPlayers)

  self:reset()
end

function seqstate:reset()
  self.playerBet:fill(1)
  self.playerFolded:zero()
  self.deadCards:zero()
  self.playerHoldings[1] = sampleCard(self.deadCards)
  self.playerHoldings[2] = sampleCard(self.deadCards)
  self.round = PREFLOP
  self.board = 0
  self.currentBet = 0
  self.currentBetTotal = ANTE
  self.potsize = self.nPlayers * ANTE
  self.playersTurn = 1
  self.terminal = false
  self.history:reset()
  self:update_observation()
end

function seqstate:clone()
  local clone = seqstate()
  clone.playerBet = self.playerBet:clone()
  clone.playerFolded = self.playerFolded:clone()
  clone.deadCards = self.deadCards:clone()
  clone.playerHoldings = self.playerHoldings:clone()
  clone.round = self.round
  clone.board = self.board
  clone.currentBet = self.currentBet
  clone.currentBetTotal = self.currentBetTotal
  clone.potsize = self.potsize
  clone.playersTurn = self.playersTurn
  clone.terminal = self.terminal
  clone.history = self.history:clone()
  clone:update_observation()
  return clone
end

function seqstate:num_players()
    return self.nPlayers
end

function seqstate:player()
  return self.playersTurn
end

function seqstate:observe()
  return self.obs
end

function seqstate:print()
  if self.terminal then
    print("Terminal state")
  elseif self.round > PREFLOP then
    print("Player "..self.playersTurn..", Round "..self.round..", Holding "..self.playerHoldings[self.playersTurn]..", Board "..self.board)
  else
    print("Player "..self.playersTurn..", Round "..self.round..", Holding "..self.playerHoldings[self.playersTurn])
  end
  self:update_observation()
  print("Card Obs:")
  print(self.obs:sub(1,6))
  print("History Obs:")
  -- print(self.obs:sub(7,28):view(2, 11))  -- seqhu
  print(self.obs:sub(7,30):view(2, 2, 3, 2)) -- hulevel
  -- print("History: ")
  -- print(self.history.hist)
end

function seqstate:update_observation()
  self.obs:zero()
  self.obs[self.playerHoldings[self.playersTurn]] = 1
  if self.round == FLOP then
    self.obs[3+self.board] = 1
  end
  local hist = self.history:observe()
  self.obs_hist:copy(hist)
  return self.obs
end

local simulator = class('LeducSimulator')

function simulator:__init(args)
end

function simulator:actionDim()
  return 3
end

function simulator:stateDim()
  return self:new_state():observe():size(1)
end

function simulator:new_state(obs)
  return seqstate(obs)
end

local function correctAction(state, action)
	if action == CHECKCALL then
    -- CHECKCALL always valid
		return action
	elseif action == FOLD then
		if state.currentBet > 0 then
      -- FOLD valid if facing a bet in Leduc
			return action
		else
      -- if nothing was bet, correct FOLD -> CHECKCALL
			return CHECKCALL
		end
	else
		local betsize = SMALL_BET
		if state.round > PREFLOP then
			betsize = BIG_BET
		end
		if state.currentBet > 1.5*betsize then
      -- if facing 2 * betsize then reached cap
			return CHECKCALL
		else
      -- currentBet has not reached cap, so can raise
			return action
		end
	end
end

function simulator:isLegal(state, action)
  return correctAction(state, action) == action
end

local function chanceMove(state)
  if state.round ~= PREFLOP then
    error("Leduc chance moves only for PREFLOP -> FLOP transition")
  end
  state.board = sampleCard(state.deadCards)
  state.round = FLOP
  state.playersTurn = 1
  state.currentBet = 0
  state.history:inc_round()
end

local function playerFold(state, rewards)
  --print("FOLD")
  state.playerFolded[state.playersTurn] = 1
  state.terminal = true
end

local function playerCheckCall(state, rewards)
  if state.currentBet == 0 then
    -- CHECK
    state.history:add(state.playersTurn, CHECK)
    state.playersTurn = state.playersTurn + 1
    if state.playersTurn > state.nPlayers then
      -- next round
      if state.round == PREFLOP then
        chanceMove(state)
      else
        state.terminal = true
      end
    end
  else
    -- CALL
    state.history:add(state.playersTurn, CALL)
    local player = state.playersTurn
    local ca = state.currentBetTotal - state.playerBet[player]
    state.playerBet[player] = state.currentBetTotal
    state.potsize = state.potsize + ca
    rewards[player] = rewards[player] - ca
    if state.round == PREFLOP then
      chanceMove(state)
    else
      state.terminal = true
    end
  end
end

local function playerBetRaise(state, rewards)
  local player = state.playersTurn
  local ba = (state.round == PREFLOP and SMALL_BET) or BIG_BET
  if state.currentBet == 0 then
    -- BET
    state.history:add(player, BET)
    state.playerBet[player] = state.playerBet[player] + ba
    state.potsize = state.potsize + ba
    state.currentBet = state.currentBet + ba
    state.currentBetTotal = state.currentBetTotal + ba
    rewards[player] = rewards[player] - ba
    state.playersTurn = state.playersTurn % state.nPlayers + 1
  else
    -- RAISE
    state.history:add(player, RAISE)
    local ca = state.currentBetTotal - state.playerBet[player]
    local ra = ca + ba
    state.playerBet[player] = state.playerBet[player] + ra
    state.potsize = state.potsize + ra
    state.currentBet = state.currentBet + ba
    state.currentBetTotal = state.currentBetTotal + ba
    rewards[player] = rewards[player] - ra
    state.playersTurn = state.playersTurn % state.nPlayers + 1
  end
  if state.playersTurn < player then
    state.history:inc_orbit()
  end
end

function simulator:step(state, action, rewards)
  rewards:zero()

  action = correctAction(state, action)

  if action == FOLD then
    playerFold(state, rewards)
  elseif action == CHECKCALL then
		playerCheckCall(state, rewards)
	elseif action == BETRAISE then
		playerBetRaise(state, rewards)
	else
		error("Invalid action passed to simulator")
	end
  -- rewards:zero()  -- using absolute terminal-only rewards
  if state.terminal then
    -- compute terminal rewards
    self:compute_terminal_rewards(state, rewards)
    -- simulator.compute_terminal_absolute_rewards(state, rewards)
  end
  rewards:mul(REW_SCALE)

  return action
end

local function computeHandStrength(holding, board)
  if holding == board then
    return holding + 3
  else
    return holding
  end
end

function simulator:compute_terminal_rewards(state, rewards)
  -- assuming 2-player Leduc
  winningPlayer = -1
  if state.playerFolded[1] == 1 then
    winningPlayer = 2
  elseif state.playerFolded[2] == 1 then
    winningPlayer = 1
  else
    -- always showdown (with board), since assuming terminal
    local p1_hstr = computeHandStrength(state.playerHoldings[1], state.board)
    local p2_hstr = computeHandStrength(state.playerHoldings[2], state.board)
    if p1_hstr > p2_hstr then
      winningPlayer = 1
    elseif p2_hstr > p1_hstr then
      winningPlayer = 2
    else
      winningPlayer = 0 -- split
    end
  end
  if winningPlayer == 0 then
    -- split
    local split = state.potsize * 0.5
    rewards[1] = rewards[1] + split
    rewards[2] = rewards[2] + split
  else
    rewards[winningPlayer] = rewards[winningPlayer] + state.potsize -- using relative rewards, thus whole pot is won
  end
end

function simulator:compute_terminal_absolute_rewards(state, rewards)
  assert(state.terminal)
  rewards:copy(state.playerBet):mul(-1)
  self:compute_terminal_rewards(state, rewards)
end

return {Leduc = simulator}
