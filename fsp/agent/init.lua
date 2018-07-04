local tablex = require 'pl.tablex'

local agents = {}

tablex.update(agents, require 'fsp.agent.fsp')

return agents
