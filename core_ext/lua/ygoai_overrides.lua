-- Loaded right after utility.lua (see ygoai/core.py): faster versions of helpers
-- that give exactly the same results.

-- Auxiliary.SelectUnselectLoop searches for a group of minc..maxc cards that
-- satisfies rescon by adding one card at a time, trying every remaining card
-- at every step: it reaches a group of k cards in up to k! orders, and a search
-- that fails visits them all (e.g. Rainbow Dark Dragon checking for 7 DARK
-- monsters with different names in a big GY took seconds at every idle prompt).
-- Whether the cards still to add can complete the group depends only on which
-- cards are in it already (rescon sees the group and the card just added), so
-- that part is remembered per group for the duration of one top-level call.
local memo = nil

local function group_key(sg)
	local ids = {}
	for tc in sg:Iter() do
		ids[#ids + 1] = tc:GetCardID()
	end
	table.sort(ids)
	return table.concat(ids, ",")
end

local loop

-- (the original's mg:IsExists(Auxiliary.SelectUnselectLoop,1,sg,sg,mg,...))
local function extends(sg, mg, e, tp, minc, maxc, rescon)
	local key = group_key(sg)
	local res = memo[key]
	if res == nil then
		res = mg:IsExists(loop, 1, sg, sg, mg, e, tp, minc, maxc, rescon)
		memo[key] = res
	end
	return res
end

loop = function(c, sg, mg, e, tp, minc, maxc, rescon)
	local res = not rescon
	if #sg >= maxc then return false end
	sg:AddCard(c)
	if rescon then
		local stop
		res, stop = rescon(sg, e, tp, mg, c)
		if stop then
			sg:RemoveCard(c)
			return false
		end
	end
	if #sg < minc then
		res = extends(sg, mg, e, tp, minc, maxc, rescon)
	elseif #sg < maxc and not res then
		res = extends(sg, mg, e, tp, minc, maxc, rescon)
	end
	sg:RemoveCard(c)
	return res
end

function Auxiliary.SelectUnselectLoop(c, sg, mg, e, tp, minc, maxc, rescon)
	if memo ~= nil then
		return loop(c, sg, mg, e, tp, minc, maxc, rescon)
	end
	-- (a top-level call: mg, e, tp, minc, maxc and rescon stay the same below it)
	memo = {}
	local ok, res = pcall(loop, c, sg, mg, e, tp, minc, maxc, rescon)
	memo = nil
	if not ok then error(res, 0) end
	return res
end
