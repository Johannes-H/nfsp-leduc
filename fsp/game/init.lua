local tablex = require 'pl.tablex'

local games = {}

tablex.update(games, require 'fsp.game.light_zerosum')

return games
