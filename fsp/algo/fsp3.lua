local memory = require 'fsp.memory'
local data = require 'fsp.data'
local learn = require 'fsp.learner'
local class = require 'class'
local optim = require 'optim'
local util = require 'fsp.util'
local fsp = class('FSP3')

function fsp:__init(params)
  -- logging
  self.logger = optim.Logger(params.log_file)
  self.plot_eval = params.plot_eval or false
  self.save_file = params.save_file
  -- policies and learners
  self.nPlayers = params.nPlayers
  self.dataGen = params.dataGen
  self.rl_memories = params.rl_memories
  self.sl_memories = params.sl_memories
  self.br_policies = params.br_policies
  self.avg_policies = params.avg_policies
  self.br_learners = params.br_learners
  self.avg_learners = params.avg_learners
  self.evaluator = params.evaluator
  -- dynamic params
  self.iter_counter = 0
  -- params
  self.gpu = params.gpu or (params.gpu == nil and true)
  self.eval_freq = params.eval_freq
  self.save_freq = params.save_freq
  --self.burnIn = params.burnIn
  self.minReplayData = params.minReplayData
  self.budget_train_br = params.budget_train_br
  self.budget_train_avg = params.budget_train_avg
  self.anticipatory_min = params.anticipatory_min
  self.anticipatory_base = params.anticipatory_base
  self.anticipatory_const = params.anticipatory_const
  self.br_lr_base = params.lr_base.br
  self.br_lr_const = params.lr_const.br
  self.avg_lr_base = params.lr_base.avg
  self.avg_lr_const = params.lr_const.avg
  self._br_lr = self.br_lr_base
  self._avg_lr = self.avg_lr_base
  self.explo_base = params.explo_base
  self.explo_const = params.explo_const
  -- misc
  self.sl_mem_reset = false
  self.rl_buffer = memory.createTransitionBuffer(params.rl_minibatch_size, self.dataGen:stateDim(), self.gpu)
  self.sl_buffer = memory.createActionBuffer(params.sl_minibatch_size, self.dataGen:stateDim(), self.gpu)
end

function fsp:run(nIter)
  local evaluation
  self.iter_counter = 1
  for i=1,nIter do

    self:iteration()
    self.iter_counter = self.iter_counter + 1
    if self.iter_counter % self.eval_freq == 0 then
      self:evaluate()
      local values, styles = self.evaluator(self.br_policies, self.avg_policies)
      print("Iteration "..self.iter_counter..", Evaluation:")
      print(values)
      self.logger:add(values)
      if self.plot_eval then
        self.logger:style(styles)
        self.logger:plot()
      end
      self:check_save()
    end
  end
end

function fsp:check_save()
  if self.save_freq >= 0 then
    if self.save_freq == 0 then
      self:save(self.save_file..".th7")
    else
      local num_evals = self.iter_counter / self.eval_freq
      if num_evals % self.save_freq == 0 then
        self:save(self.save_file.."_"..num_evals..".th7")
      end
    end
  end
end

function fsp:update_scheduled_params()
  self.anticipatory = math.max(self.anticipatory_min, util.polynomialDecay(self.anticipatory_base, self.anticipatory_const, 0.5, self.iter_counter))  -- == learning rate of FP
  self._br_lr = util.polynomialDecay(self.br_lr_base, self.br_lr_const, 0.5, self.iter_counter)
  self._avg_lr = util.linearDecay(self.avg_lr_base, self.avg_lr_const, self.iter_counter)
  local explo = util.polynomialDecay(self.explo_base, self.explo_const, 0.5, self.iter_counter)
  for p=1,self.nPlayers do
    self.dataGen:setExploration(explo)
    self.dataGen:setAnticipatory(self.anticipatory)
    self.br_learners[p].optim_state.learningRate = self._br_lr
    self.avg_learners[p].optim_state.learningRate = self._avg_lr
  end
  if self.iter_counter % self.eval_freq == 0 then
    print("Using scheduled params: br_lr "..self._br_lr..", avg_lr "..self._avg_lr..", anticipatory "..self.anticipatory..", exploration "..explo)
  end
end

function fsp:save(filename)
  local to_save = {
    br = self.br_policies,
    avg = self.avg_policies,
  }
  torch.save(filename, to_save)
end

function fsp:training()
  for p=1,self.nPlayers do
    self.br_policies[p].net:training()
    self.avg_policies[p].net:training()
  end
end

function fsp:evaluate()
  for p=1,self.nPlayers do
    self.br_policies[p].net:evaluate()
    self.avg_policies[p].net:evaluate()
  end
end

function fsp:iteration()
  self:update_scheduled_params()
  self:generate_experience()
  self:replay_experience()

end

function fsp:generate_experience()
  self:evaluate()
  self.dataGen:generate()
end

function fsp:replay_experience()
  self:training()
  self:train_br()
  self:train_avg()
end

function fsp:train_br()
  for i=1,self.nPlayers do
    if self.rl_memories[i]:length() > self.minReplayData then
      if self.iter_counter % self.eval_freq == 0 then
        print("Player "..i.." training its BR on memory of length "..self.rl_memories[i]:length())
      end
      learn.Learn(self.budget_train_br, self.rl_buffer, self.br_learners[i], self.rl_memories[i])
    end
  end
end

function fsp:train_avg()
  for i=1,self.nPlayers do
    if self.sl_memories[i]:length() > self.minReplayData then
      if self.iter_counter % self.eval_freq == 0 then
        print("Player "..i.." training its AVG on memory of length "..self.sl_memories[i]:length())
      end
      learn.Learn(self.budget_train_avg, self.sl_buffer, self.avg_learners[i], self.sl_memories[i])
    end
  end
end

return {FSP3 = fsp}
