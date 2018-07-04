local class = require 'class'
local base = require 'fsp.learner.learner'
local learner = class('QLearner', 'Learner')

function learner:__init(params)
  base.Learner.__init(self, params)
  self.target_net_refresh = params.target_net_refresh
  self.target_net_counter = 0
  self.output_grad = torch.FloatTensor()
  self:update_target_net()
  self.gpu = params.gpu
end

function learner:learn(buffer)
  local o_tm1, a_tm1, r_t, o_t, c_t = buffer.o_tm1, buffer.a_tm1, buffer.r_t, buffer.o_t, buffer.c_t
  if self.target_net_counter % self.target_net_refresh == 0 then
    self:update_target_net()
  end
  self.target_net_counter = self.target_net_counter + 1

  self.target_net:evaluate()
  local q_t_max = self.target_net:forward(o_t):float():max(2):cmul(c_t)
  self.target_net:training()
  local td_target = r_t:add(q_t_max)
  -- accumulate td error
  local q_tm1 = self.net:forward(o_tm1):float()
  local qDim = q_tm1:size()
  self.output_grad:resizeAs(q_tm1):zero()
  local cum_td_err = 0
  local part_td_err
  local nBatch = qDim[1]
  for i=1, nBatch do
    part_td_err = q_tm1[i][a_tm1[i]] - td_target[i]
    self.output_grad[i][a_tm1[i]] = part_td_err -- store err/criterion derivative
    cum_td_err = cum_td_err + part_td_err * part_td_err -- accumulate total error
  end
  self.output_grad:div(nBatch)
  cum_td_err = cum_td_err / nBatch
  if self.gpu then
    self.net:backward(o_tm1, self.output_grad:cuda())
  else
    self.net:backward(o_tm1, self.output_grad)
  end
  --cutorch.synchronize()
  self.optim_err = self.optim_err + cum_td_err
end

function learner:update_target_net()
  if self.target_net_refresh > 0 then
    self.target_net = self.net:clone()
    self.target_net:evaluate()
  else
    self.target_net = self.net  -- dont fit target net
  end
end

return {Q = learner}
