
local function getSqErr(probs, target)
  local diff = probs[2] - target  -- target always action 2
  return 2*diff*diff  -- 2* due to symmetry of both actions
end

local function evalP2(avg_policy, potsize)
  local input = torch.CudaTensor(6):fill(0)
  -- doing p1 check first
  print("P2 vs P1 check")
  input[4] = 1 -- p1 check
  input[1] = 1 -- J
  local probs = avg_policy:probs(input)
  print(probs)
  local sq_err = 0
  sq_err = sq_err + getSqErr(probs, 1/(potsize+1))
  input[1] = 0
  input[2] = 1  -- Q
  probs = avg_policy:probs(input)
  print(probs)
  sq_err = sq_err + getSqErr(probs, 0)
  input[2] = 0
  input[3] = 1  -- K
  probs = avg_policy:probs(input)
  print(probs)
  sq_err = sq_err + getSqErr(probs, 1)
  -- doing p1 bet
  print("P2 vs P1 bet")
  input:zero()
  input[5] = 1  -- p1 bet
  input[1] = 1 -- J
  probs = avg_policy:probs(input)
  print(probs)
  sq_err = sq_err + getSqErr(probs, 0)
  input[1] = 0
  input[2] = 1 -- Q
  probs = avg_policy:probs(input)
  print(probs)
  sq_err = sq_err + getSqErr(probs, (potsize-1)/(potsize+1))
  input[2] = 0
  input[3] = 1 -- K
  probs = avg_policy:probs(input)
  print(probs)
  sq_err = sq_err + getSqErr(probs, 1)
  return sq_err
end

local function determineClosestKuhnNash(alpha, beta, gamma, potsize)
  local nash = torch.Tensor(3)
  nash[3] = (potsize+1) * (alpha+beta+(potsize+1)*gamma) + 1 - potsize
  nash[3] = nash[3] / ((potsize+1)*(potsize+1)+2)
  nash[1] = nash[3]/(1+potsize)
  nash[2] = (potsize-1+nash[3])/(1+potsize)
  return nash
end

local function evalP1(avg_policy, potsize)
  local alpha, beta, gamma
  local input = torch.CudaTensor(6):fill(0)
  input[1] = 1 -- J
  local probs = avg_policy:probs(input)
  alpha = probs[2]
  input[1] = 0
  input[3] = 1  -- K
  probs = avg_policy:probs(input)
  gamma = probs[2]
  input[3] = 0
  input[2] = 1
  input[4] = 1  -- p1 check
  input[6] = 1  -- p2 bet
  probs = avg_policy:probs(input)
  beta = probs[2]
  nash = determineClosestKuhnNash(alpha, beta, gamma, potsize)
  input:zero()
  input[2] = 1 -- Q
  probs = avg_policy:probs(input)
  local targetPolicy = torch.Tensor(3)
  targetPolicy[1] = nash[1]
  targetPolicy[2] = 0
  targetPolicy[3] = nash[3]
  local netPolicy = torch.Tensor(3)
  netPolicy[1] = alpha
  netPolicy[2] = probs[2]
  netPolicy[3] = gamma
  netPolicy:add(-1, targetPolicy)
  netPolicy:cmul(netPolicy)
  local sq_err = netPolicy:sum()*2
  targetPolicy[1] = 0
  targetPolicy[2] = nash[2]
  targetPolicy[3] = 1
  input:zero()
  input[4] = 1  -- p1 check
  input[6] = 1  -- p2 bet
  input[1] = 1
  probs = avg_policy:probs(input)
  netPolicy[1] = probs[2]
  netPolicy[2] = beta
  input[1] = 0
  input[3] = 1
  probs = avg_policy:probs(input)
  netPolicy[3] = probs[2]
  netPolicy:add(-1, targetPolicy)
  netPolicy:cmul(netPolicy)
  sq_err = sq_err + netPolicy:sum()*2
  return sq_err
end


local function printNet(input, br, avg)
  print("Input:")
  print(input)
  local output = br:forward(input)
  print("BR (Q values):")
  print(output)
  output = avg:forward(input)
  print("Avg:")
  print(output)
end

local function eval(br_policies, avg_policies)
  local input = torch.CudaTensor(6):fill(0)
  input[4] = 1  -- p1 bet
  input[2] = 1  -- p2 Q

  printNet(input, br_policies[2].net, avg_policies[2].net)
  local inputs = {torch.CudaTensor(6):fill(0)}
  inputs[2] = inputs[1]:clone()
  inputs[3] = inputs[1]:clone()
  inputs[1][1] = 1
  inputs[2][2] = 1
  inputs[3][3] = 1
  printNet(inputs[1], br_policies[1].net, avg_policies[1].net)
  printNet(inputs[2], br_policies[1].net, avg_policies[1].net)
  printNet(inputs[3], br_policies[1].net, avg_policies[1].net)
  inputs[1][4] = 1
  inputs[1][6] = 1
  printNet(inputs[1], br_policies[1].net, avg_policies[1].net)
  inputs[2][4] = 1
  inputs[2][6] = 1
  printNet(inputs[2], br_policies[1].net, avg_policies[1].net)
  inputs[3][4] = 1
  inputs[3][6] = 1
  printNet(inputs[3], br_policies[1].net, avg_policies[1].net)
  local sq_err_p1, sq_err_p2
  avg_sq_err_p1 = evalP1(avg_policies[1], 2)
  avg_sq_err_p2 = evalP2(avg_policies[2], 2)
  print("sq_err_p1 "..avg_sq_err_p1)
  print("sq_err_p2 "..avg_sq_err_p2)
  local avg_sq_err = avg_sq_err_p1 + avg_sq_err_p2
  local br_sq_err = evalP1(br_policies[1], 2) + evalP2(br_policies[2], 2)

  local values = {['avg_sq_err'] = avg_sq_err, ['br_sq_err'] = br_sq_err}
  local styles = {['avg_sq_err'] = '-', ['br_sq_err'] = '-'}
  return values, styles
end

return eval
