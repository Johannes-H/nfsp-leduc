local tablex = require 'pl.tablex'

local learners = {}

tablex.update(learners, require 'fsp.learner.qlearner')
tablex.update(learners, require 'fsp.learner.avglearner')

learners.Learn = function(nSteps, buffer, learner, memory)
  for i=1,nSteps do
    memory:fill(buffer)
    learner:learn(buffer)
    learner:adapt()
  end
end

return learners
