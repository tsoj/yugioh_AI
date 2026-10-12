--宣告者の神託
--Oracle of the Herald
-- (ygoai: the original also called Duel.SetChainLimit when the activation was only
-- checked (chk==0) and could not be activated: a set copy then kept the opponent
-- from chaining to unrelated events, e.g. an attack declaration)
local s,id=GetID()
function s.initial_effect(c)
	local e1=Ritual.AddProcGreaterCode(c,12,nil,48546368)
	e1:SetTarget(s.target(e1))
end
function s.target(eff)
	local tg = eff:GetTarget()
	return function(e,tp,eg,ep,ev,re,r,rp,chk,...)
		local ret = tg(e,tp,eg,ep,ev,re,r,rp,chk,...)
		if chk==0 then return ret end
		if e:IsHasType(EFFECT_TYPE_ACTIVATE) then
			Duel.SetChainLimit(s.chlimit)
		end
		return ret
	end
end
function s.chlimit(e,ep,tp)
	return tp==ep
end
