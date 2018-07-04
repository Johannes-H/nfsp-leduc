local class = require 'class'

local greedy = class('GreedyPolicy')

function greedy:__init(params)
  self.net = params.net
  self.actionDim = params.actionDim
end

function greedy:evaluate()
  self.net:evaluate()
end

function greedy:training()
  self.net:training()
end

function greedy:action(obs)
  local q = self.net:forward(obs:cuda()):float()
  local _, maxIdT = torch.max(q,1)
  return maxIdT[1]
end

function greedy:actions(batch)
  local _, maxIdT = torch.max(self.net:forward(batch):float(), 2)
  return maxIdT:select(2,1)
end

function greedy:probs(batch)
  if batch:dim() > 1 then
    local nBatch = batch:size(1)
    local probs = torch.Tensor(nBatch, self.actionDim):zero()
    local actions = self:actions(batch)
    for i=1,nBatch do
      probs[i][actions[i]] = 1
    end
    return probs
  else
    local probs = torch.Tensor(self.actionDim):zero()
    probs[self:action(batch)] = 1
    return probs
  end
end

function greedy:setExploration(explo)
  error("Greedy policy does not have an exploration term")
end


local epsgreedy = class('EpsGreedyPolicy')

function epsgreedy:__init(params)
  self.net = params.net
  self.epsilon = params.epsilon
  self.actionDim = params.actionDim
end

function epsgreedy:evaluate()
  self.net:evaluate()
end

function epsgreedy:training()
  self.net:training()
end

local function sampleEpsGreedy(q, eps, actionDim)
  if torch.FloatTensor.torch.uniform() < eps then
    return torch.FloatTensor.torch.random(actionDim)
  else
    local _, maxIdT = torch.max(q,1)
    return maxIdT[1]
  end
end

function epsgreedy:action(obs)
  if torch.FloatTensor.torch.uniform() < self.epsilon then
    return torch.FloatTensor.torch.random(self.actionDim)
  else
    local q = self.net:forward(obs):float()
    local _, maxIdT = torch.max(q,1)
    return maxIdT[1]
  end
end

function epsgreedy:actions(batch)
  local q = self.net:forward(batch):float()
  local nBatch = batch:size(1)
  local actions = torch.LongTensor(nBatch)
  for i=1,nBatch do
    actions[i] = sampleEpsGreedy(q[i], self.epsilon, self.actionDim)
  end
  return actions
end

function epsgreedy:probs(batch)
  if batch:dim() > 1 then
    local q = self.net:forward(batch:cuda()):float()
    local nBatch = batch:size(1)
    local nonGreedy = self.epsilon/self.actionDim
    local greedy = 1-self.epsilon + nonGreedy
    local probs = torch.Tensor(nBatch, self.actionDim):fill(nonGreedy)
    local _, maxIdT = torch.max(q, 2)
    for i=1,nBatch do
      probs[i][maxIdT[i][1]] = greedy
    end
    return probs
  else
    local probs = torch.Tensor(self.actionDim):fill(self.epsilon/self.actionDim)
    local q = self.net:forward(batch:cuda()):float()
    local _, maxIdT = torch.max(q,1)
    probs[maxIdT[1]] = probs[maxIdT[1]] + 1 - self.epsilon
    return probs
  end
end

function epsgreedy:setExploration(explo)
  self.epsilon = explo
end



local light_epsgreedy = class('LightEpsGreedyPolicy')

function light_epsgreedy:__init(params)
  self.batch_net = params.net -- network that runs forward batches for multiple policies
  self.policy_no = params.policy_no -- no of this policy, to pick correct row of output

  self.epsilon = params.epsilon
  self.actionDim = params.actionDim
end

function light_epsgreedy:light_clone(policy_no)
  return light_epsgreedy{net = self.batch_net, policy_no = policy_no, epsilon = self.epsilon, actionDim = self.actionDim}
end

function light_epsgreedy:action()
  if torch.FloatTensor.torch.uniform() < self.epsilon then
    return torch.FloatTensor.torch.random(self.actionDim)
  else
    local _, maxIdT = self.batch_net.output[self.policy_no]:max(1)
    return maxIdT[1]
  end
end

function light_epsgreedy:probs()
  self._probs = self.batch_net.output[self.policy_no] -- inplace, replacing Q values as probs
  local _, maxIdT = self._probs:max(1)
  self._probs:fill(self.epsilon/self.actionDim)
  self._probs[maxIdT[1]] = 1-self.epsilon + self.epsilon/self.actionDim
  return self._probs
end

function light_epsgreedy:setExploration(explo)
  self.epsilon = explo
end


local light_greedy = class('LightGreedyPolicy')

function light_greedy:__init(params)
  self.batch_net = params.net -- network that runs forward batches for multiple policies
  self.policy_no = params.policy_no -- no of this policy, to pick correct row of output
end

function light_greedy:light_clone(policy_no)
  return light_greedy{net = self.batch_net, policy_no = policy_no}
end

function light_greedy:action()
  local _, maxIdT = self.batch_net.output[self.policy_no]:max(1)
  return maxIdT[1]
end

function light_greedy:setExploration(explo)
end

return {Greedy = greedy, EpsGreedy = epsgreedy, LightEpsGreedy = light_epsgreedy, LightGreedy = light_greedy}
