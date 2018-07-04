local tablex = require 'pl.tablex'

local simulators = {}

-- zero-sum games
tablex.update(simulators, require 'fsp.simulator.kuhn')
tablex.update(simulators, require 'fsp.simulator.leduc')
tablex.update(simulators, require 'fsp.simulator.leduc_eval')
simulators.KuhnEval = require 'fsp.simulator.kuhn_eval'
simulators.History = require 'fsp.simulator.history'


return simulators
