
local simulator = require 'fsp.simulator.leduc'.Leduc()

local NUM_HOLDINGS = 3
local NUM_BOARD = 3

local FOLD = 1
local CHECKCALL = 2
local BETRAISE = 3

local ACTION_CHAR = {"f", "c", "r"}

local function extractQs(state, net, num_board_cards)
  local qs = torch.DoubleTensor(num_board_cards, NUM_HOLDINGS, 3)
  state:update_observation()
  local stateObs = state:observe()
  local batchObs = torch.CudaTensor(1, stateObs:size(1)):copy(stateObs)
  local obs = batchObs[1]
  for b=1,num_board_cards do
    if num_board_cards > 1 then
      obs:sub(4,6):zero()
      obs[3+b] = 1
    end
    for h=1,NUM_HOLDINGS do
      obs:sub(1,3):zero()
      obs[h] = 1
      qs[b][h]:copy(net:forward(batchObs))
    end
  end
  return qs
end

local function extractProbs(state, policy, num_board_cards)
  local probs = torch.DoubleTensor(num_board_cards, NUM_HOLDINGS, 3)
  state:update_observation()
  local stateObs = state:observe()
  local batchObs = torch.Tensor(1, stateObs:size(1)):copy(stateObs)
  local obs = batchObs[1]

  for b=1,num_board_cards do
    if num_board_cards > 1 then
      obs:sub(4,6):zero()
      obs[3+b] = 1
    end
    for h=1,NUM_HOLDINGS do
      obs:sub(1,3):zero()
      obs[h] = 1
      probs[b][h]:copy(policy:probs(batchObs))
    end
  end
  if not simulator:isLegal(state, FOLD) then
    -- move fold probability to CHECK prob
    local foldProbs = probs:select(3, 1)
    probs:select(3,2):add(foldProbs)
    foldProbs:zero()
  elseif not simulator:isLegal(state, BETRAISE) then
    -- move raise probability to CALL prob
    local raiseProbs = probs:select(3, 3)
    probs:select(3,2):add(raiseProbs)
    raiseProbs:zero()
  end
  return probs
end

local function accumulateTerminalExplo(state, br_reach, avg_reach, explo)
  local no_board = (state.round == 1 and true) or false
  local num_board_cards = (no_board and 1) or NUM_BOARD

  local board_collisions, board_prob, holding_prob
  local rewards = torch.DoubleTensor(2)

  for b=1, num_board_cards do
    for h1=1, NUM_HOLDINGS do
      if h1 == b then
        board_collisions = 1
      else
        board_collisions = 0
      end
      for h2=1, NUM_HOLDINGS do
        if no_board then
          board_prob = 1
        else
          if h2 == b then
            board_collisions = board_collisions + 1
          end
          -- compute board prob
          if board_collisions == 2 then
            board_prob = 0
          elseif board_collisions == 1 then
            board_prob = 0.25
          else
            board_prob = 0.5
          end
        end
          -- compute holding prob
          if h1 == h2 then
            holding_prob = 0.2/3
          else
            holding_prob = 0.4/3
          end
          -- get absolute termianl rewards
          state.board = b
          state.playerHoldings[1] = h1
          state.playerHoldings[2] = h2
          simulator:compute_terminal_absolute_rewards(state, rewards)
          -- accumulate exploitabilities
          explo[1] = explo[1] + br_reach[1][b][h1] * avg_reach[2][b][h2] * holding_prob * board_prob * rewards[1]
          explo[2] = explo[2] + br_reach[2][b][h2] * avg_reach[1][b][h1] * holding_prob * board_prob * rewards[2]

          if h2 == b then
            board_collisions = board_collisions - 1
          end

      end
    end
  end

end


function _br_eval(br_policies, avg_policies, state, br_reach, avg_reach, explo)
  local dump_rewards = torch.Tensor(2)
  if state.terminal then
    accumulateTerminalExplo(state, br_reach, avg_reach, explo)
  else
    local player = state:player()
    local opponent = player%2 + 1
    local num_board_cards = (state.round == 1 and 1) or NUM_BOARD

    local next_br_reach = br_reach:clone()
    local next_avg_reach = avg_reach:clone()

    local br_probs = extractProbs(state, br_policies[player], num_board_cards)
    local avg_probs = extractProbs(state, avg_policies[player], num_board_cards)

    local next_state
    for a=1,3 do
      if simulator:isLegal(state, a) then
        next_state = state:clone()
        simulator:step(next_state, a, dump_rewards)
        next_br_reach:copy(br_reach)
        next_avg_reach:copy(avg_reach)
        for b=1,NUM_BOARD do
          local eff_b = (b-1)%num_board_cards+1
          for h=1,NUM_HOLDINGS do
            next_br_reach[player][b][h] = next_br_reach[player][b][h] * br_probs[eff_b][h][a]
            next_avg_reach[player][b][h] = next_avg_reach[player][b][h] * avg_probs[eff_b][h][a]
          end
        end
        _br_eval(br_policies, avg_policies, next_state, next_br_reach, next_avg_reach, explo)
      end
    end
  end
end


function _policy_compare(policies1, policies2, state, sequence, comp_thresh)
  local dump_rewards = torch.Tensor(2)
  if not state.terminal then
    local player = state:player()
    local opponent = player%2 + 1
    local num_board_cards = (state.round == 1 and 1) or NUM_BOARD

    local probs_1 = extractProbs(state, policies1[player], num_board_cards)
    local probs_2 = extractProbs(state, policies2[player], num_board_cards)

    if probs_1:add(-1, probs_2):abs():sum() > comp_thresh then
      print("Found different policies at sequence:")
      print(sequence)
      print("Policies:")
      print(probs_1)
      print(probs_2)
      print(extractQs(state, policies2[player].net, num_board_cards))
    end

    local next_state, next_sequence
    for a=1,3 do
      if simulator:isLegal(state, a) then
        next_state = state:clone()
        simulator:step(next_state, a, dump_rewards)
        next_sequence = sequence.."/"..player..ACTION_CHAR[a]
        _policy_compare(policies1, policies2, next_state, next_sequence, comp_thresh)
      end
    end
  end
end


local function extractTerminalValues(state, reach, values)
  -- TERMINAL_COUNT = TERMINAL_COUNT + 1
  values:zero()
  local no_board = (state.round == 1 and true) or false
  local board_collisions, board_prob, holding_prob
  local rewards = torch.DoubleTensor(2)

  for b=1, values:size(2) do
    for h1=1, NUM_HOLDINGS do
      if h1 == b then
        board_collisions = 1
      else
        board_collisions = 0
      end
      for h2=1, NUM_HOLDINGS do
        if no_board then
          board_prob = 1
        else
          if h2 == b then
            board_collisions = board_collisions + 1
          end
          -- compute board prob
          if board_collisions == 2 then
            board_prob = 0
          elseif board_collisions == 1 then
            board_prob = 0.25
          else
            board_prob = 0.5
          end
        end
          -- compute holding prob
          if h1 == h2 then
            holding_prob = 0.2/3
          else
            holding_prob = 0.4/3
          end
          -- get absolute termianl rewards
          state.board = b
          state.playerHoldings[1] = h1
          state.playerHoldings[2] = h2
          simulator:compute_terminal_absolute_rewards(state, rewards)
          -- accumulate values
          values[1][b][h1] = values[1][b][h1] + reach[2][b][h2] * holding_prob * board_prob * rewards[1]
          values[2][b][h2] = values[2][b][h2] + reach[1][b][h1] * holding_prob * board_prob * rewards[2]

          if h2 == b then
            board_collisions = board_collisions - 1
          end
      end
    end
  end

end

local function _eval(policies, state, reach, prev_values)
  local dump_rewards = torch.Tensor(2)
  if state.terminal then
    extractTerminalValues(state, reach, prev_values)
  else
    local player = state:player()
    local opponent = player%2 + 1
    local num_board_cards = (state.round == 1 and 1) or NUM_BOARD

    local max_player_values = torch.DoubleTensor(num_board_cards, NUM_HOLDINGS):fill(-999)
    local max_player_actions = torch.ByteTensor(num_board_cards, NUM_HOLDINGS)
    local opponent_values = torch.DoubleTensor(num_board_cards, NUM_HOLDINGS):zero()
    local next_reach = reach:clone()
    local values = torch.DoubleTensor(2, num_board_cards, NUM_HOLDINGS)

    local probs = extractProbs(state, policies[player], num_board_cards)

    local next_state
    for a=1,3 do
      if simulator:isLegal(state, a) then
        next_state = state:clone()
        simulator:step(next_state, a, dump_rewards)
        next_reach:copy(reach)
        for b=1,NUM_BOARD do
          local eff_b = (b-1)%num_board_cards+1
          for h=1,NUM_HOLDINGS do
            next_reach[player][b][h] = next_reach[player][b][h] * probs[eff_b][h][a]
          end
        end
        _eval(policies, next_state, next_reach, values)
        -- evaluate values
        for b=1,num_board_cards do
          for h=1,NUM_HOLDINGS do
            -- pick highest values/actions for player
            if max_player_values[b][h] < values[player][b][h] then
              max_player_values[b][h] = values[player][b][h]
              max_player_actions[b][h] = a
            end
            -- sum values for opponent (as already prob weighted)
            opponent_values[b][h] = opponent_values[b][h] + values[opponent][b][h]
          end
        end
      end
    end

    -- update prev_values
    prev_values:zero()
    local num_prev_board_cards = prev_values:size(2)
    for b=1,num_board_cards do
      local eff_b = (b-1)%num_prev_board_cards+1
      for h=1,NUM_HOLDINGS do
        prev_values[player][eff_b][h] = prev_values[player][eff_b][h] + max_player_values[b][h]
        prev_values[opponent][eff_b][h] = prev_values[opponent][eff_b][h] + opponent_values[b][h]
      end
    end

  end
end

local function evalExploitability(policies)
  local values = torch.DoubleTensor(2, 1, NUM_HOLDINGS)
  local reach = torch.DoubleTensor(2, NUM_BOARD, NUM_HOLDINGS):fill(1)

  _eval(policies, simulator:new_state(), reach, values)

  local exploitability = torch.DoubleTensor(2):zero()
  for p=1,2 do
    for h=1,NUM_HOLDINGS do
      exploitability[p] = exploitability[p] + values[p][1][h]
    end
  end
  return exploitability
end

local function evalBRExploitingAvg(br_policies, avg_policies, exploitability)
  local br_reach = torch.DoubleTensor(2, NUM_BOARD, NUM_HOLDINGS):fill(1)
  local avg_reach = torch.DoubleTensor(2, NUM_BOARD, NUM_HOLDINGS):fill(1)
  local explo = torch.DoubleTensor(2):zero()
  _br_eval(br_policies, avg_policies, simulator:new_state(), br_reach, avg_reach, explo)
  explo:mul(-1):add(exploitability)
  return explo
end

local function verboseEval(br_policies, avg_policies)
  local br_nets = {br_policies[1].net, br_policies[2].net}
  if br_nets[1] == nil then
    br_nets = {br_policies[1].br.net, br_policies[2].br.net}
  end
  local rewards = torch.Tensor(2)
  local state = simulator:new_state()
  simulator:step(state, 3, rewards)
  local qs = extractQs(state, br_nets[2], 1)
  local br_probs = extractProbs(state, br_policies[2], 1)
  local probs = extractProbs(state, avg_policies[2], 1)
  print("P2 vs P1 bet:")
  print(qs)
  print(br_probs)
  print(probs)
  simulator:step(state, 3, rewards)
  simulator:step(state, 2, rewards)
  simulator:step(state, 3, rewards)
  qs = extractQs(state, br_nets[2], 3)
  br_probs = extractProbs(state, br_policies[2], 3)
  probs = extractProbs(state, avg_policies[2], 3)
  print("P2 vs P1 bet/call pre, bet flop:")
  print(qs)
  print(br_probs)
  print(probs)
  simulator:step(state, 3, rewards)
  qs = extractQs(state, br_nets[1], 3)
  br_probs = extractProbs(state, br_policies[1], 3)
  probs = extractProbs(state, avg_policies[1], 3)
  print("P1 vs P2 raise pre, raise flop:")
  print(qs)
  print(br_probs)
  print(probs)
end

local function verboseEvalP1(policy)
  local net = policy.net
  local rewards = torch.Tensor(2)
  local state = simulator:new_state()
  local probs = extractProbs(state, policy, 1)
  local qs = extractQs(state, net, 1)
  print("P1 first-in:")
  print(probs)
  print(qs)
  simulator:step(state, 2, rewards)
  simulator:step(state, 3, rewards)
  probs = extractProbs(state, policy, 1)
  qs = extractQs(state, net, 1)
  print("P1 check vs P2 bet:")
  print(probs)
  print(qs)
  simulator:step(state, 2, rewards)
  probs = extractProbs(state, policy, 3)
  qs = extractQs(state, net, 3)
  print("P1 check/call vs P2 bet:")
  print(probs)
  print(qs)
  simulator:step(state, 2, rewards)
  simulator:step(state, 3, rewards)
  probs = extractProbs(state, policy, 3)
  qs = extractQs(state, net, 3)
  print("P1 check/call, check vs P2 bet, bet:")
  print(probs)
  print(qs)
  state:reset()
  simulator:step(state, 3, rewards)
  simulator:step(state, 3, rewards)
  simulator:step(state, 2, rewards)
  simulator:step(state, 3, rewards)
  simulator:step(state, 3, rewards)
  probs = extractProbs(state, policy, 3)
  print("P1 vs P2 raise pre, raise flop:")
  print(probs)
end

local function eval(br_policies, avg_policies)
  local eval_avg = evalExploitability(avg_policies)
  local br_explo_avg = evalBRExploitingAvg(br_policies, avg_policies, eval_avg)
  local values = {['exploitability'] = eval_avg:mean(), ['exploitability P1'] = eval_avg[1], ['exploitability P2'] = eval_avg[2], ['br explo gap P1'] = br_explo_avg[1], ['br explo gap P2'] = br_explo_avg[2]}
  local styles = {['exploitability'] = '-', ['exploitability P1'] = '-', ['exploitability P2'] = '-', ['br explo gap P1'] = '-', ['br explo gap P2'] = '-'}
  -- verboseEval(br_policies, avg_policies)
  return values, styles
end

local function comparePolicies(policies1, policies2, comp_thresh)
  _policy_compare(policies1, policies2, simulator.new_state(), "", comp_thresh)
end

local function evalPlayerPolicy(policies, player, player_name)
  player_name = player_name or "player"
  local eval_avg = evalExploitability(policies)
  local br_explo_avg = evalBRExploitingAvg(policies, policies, eval_avg)
  local values = {['opponent exploitability'] = eval_avg[player], [player_name..' br explo gap'] = br_explo_avg[player]}
  local styles = {['opponent exploitability'] = '-', [player_name..' br explo gap'] = '-'}
  if player == 1 then
    print("########000########\nVerbose eval of "..player_name.."\n#############################")
    verboseEvalP1(policies[player])
  end
  return values, styles
end

return {LeducEval = eval, LeducExploitability = evalExploitability, LeducCompare = comparePolicies, LeducPlayerEval = evalPlayerPolicy}
