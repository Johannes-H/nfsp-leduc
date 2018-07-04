local tablex = require 'pl.tablex'

local policies = {}

tablex.update(policies, require 'fsp.policy.greedy')
tablex.update(policies, require 'fsp.policy.network')

return policies
