local class = require 'class'
local learner = class('Learner')
local optim = require 'optim'

function learner:__init(params)
  self.net = params.net
  self.params, self.grad_params = self.net:getParameters()
  self.grad_params:zero()
  self.optim_method = params.optim_method or optim.sgd
  self.optim_state = params.optim_state or {learningRate = 0.01}
  self.optim_err = 0
end

function learner:adapt()
  local feval = function()
    return self.optim_err, self.grad_params
  end
  self.optim_method(feval, self.params, self.optim_state)
  self.optim_err = 0
  self.grad_params:zero()
end

return {Learner = learner}
