local class = require 'class'
local base = require 'fsp.learner.learner'

local actions_learner = class('AvgActionsLearner', 'Learner')

function actions_learner:__init(params)
  base.Learner.__init(self, params)
  self.output_err = torch.FloatTensor()
  self.epsilon = params.epsilon or 1e-8
  self.gpu = params.gpu
end

function actions_learner:learn(buffer)
  local obs, actions = buffer.obs, buffer.actions
  local output = self.net:forward(obs):float():add(self.epsilon)
  local log_output = torch.log(output)
  local neg_err = 0
  self.output_err:resizeAs(output):zero()
  local nBatch = actions:size(1)
  for i=1,nBatch do
    neg_err = neg_err + log_output[i][actions[i]]
    self.output_err[i][actions[i]] = -1/output[i][actions[i]]
  end
  self.output_err:div(nBatch)
  neg_err = neg_err/nBatch
  if self.gpu then
    self.net:backward(obs, self.output_err:cuda())
  else
    self.net:backward(obs, self.output_err)
  end
  self.optim_err = self.optim_err - neg_err
end

return {AvgActions = actions_learner}
