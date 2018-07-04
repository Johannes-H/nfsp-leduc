local class = require 'class'

local huhu_level = class('HULevelHistory')

local compactAction = {0, 1, 1, 2, 2}	--fold, check, call, bet, raise

function huhu_level:__init(params)
	self.nPlayers = 2
  self.numRounds = params.numRounds or 2
	self.numRaises = params.numRaises or 2
	local numLevels = self.numRaises+1
	local numActions = 2
  self.hist = torch.ByteTensor(self.nPlayers, self.numRounds, numLevels, numActions)
  self.histFlat = self.hist:view(self.nPlayers * self.numRounds * numLevels * numActions)
	self:reset()
end

function huhu_level:reset()
  self.level = 1
  self.round = 1
  self.hist:zero()
end

function huhu_level:clone()
  local clone = huhu_level{numRounds = self.numRounds, numRaises = self.numRaises}
  clone.hist:copy(self.hist)
  clone.level = self.level
  clone.round = self.round
  return clone
end

function huhu_level:add(player, action)
  if action > 1 then
    -- ignoring FOLD in HUHU history
		local compact = compactAction[action]
	  self.hist[player][self.round][self.level][compact] = 1
		if compact == 2 then
			self.level = self.level+1
		end
  end
end

function huhu_level:inc_orbit()
end

function huhu_level:inc_round()
  self.round = self.round + 1
  self.level = 1
end

function huhu_level:observe()
	return self.histFlat
end


return {HULevel = huhu_level}
