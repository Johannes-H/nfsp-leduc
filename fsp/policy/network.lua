local class = require 'class'
local policy = class('NetworkPolicy')
local util = require 'fsp.util'

function policy:__init(params)
  self.net = params.net
end

function policy:evaluate()
  self.net:evaluate()
end

function policy:training()
  self.net:training()
end

function policy:action(obs)
  return util.sample(self:probs(obs:cuda()))
end

function policy:actions(batch)
  return util.sampleActions(self:probs(batch))
end

function policy:probs(batch)
  return self.net:forward(batch)
end



local epsilon = class('EpsilonNetworkPolicy', 'NetworkPolicy')

function epsilon:__init(params)
  policy.__init(self, params)
  self.actionDim = params.actionDim
  self.epsilon = params.epsilon
end

function epsilon:probs(batch)
  return self.net:forward(batch):mul(1-self.epsilon):add(self.epsilon/self.actionDim)
end



local light_policy = class('LightNetworkPolicy')

function light_policy:__init(params)
  self.batch_net = params.net -- network that runs forward batches for multiple policies
  self.policy_no = params.policy_no -- no of this policy, to pick correct row of output
end

function light_policy:light_clone(policy_no)
  return light_policy{net = self.batch_net, policy_no = policy_no}
end

function light_policy:probs()
  return self.batch_net.output[self.policy_no]
end

function light_policy:action()
  return util.sample(self:probs())
end

function light_policy:setExploration(explo)
end


return {Network = policy, EpsilonNetwork = epsilon, LightNetwork = light_policy}
