--無効
--Muko
-- (ygoai: s.repop replaces the draw effect's operation, so its ev is that effect's event value, not its chain link;
-- GetOperationInfo(ev,...) then read another chain link (or none, and cp<2 raised a Lua error on nil); the fix reads
-- chain link 0, the resolving draw effect)
local s,id=GetID()
function s.initial_effect(c)
	--change effect
	local e1=Effect.CreateEffect(c)
	e1:SetType(EFFECT_TYPE_ACTIVATE)
	e1:SetCode(EVENT_CHAINING)
	e1:SetCondition(s.condition)
	e1:SetOperation(s.activate)
	c:RegisterEffect(e1)
end
function s.condition(e,tp,eg,ep,ev,re,r,rp)
	if not re:IsHasProperty(EFFECT_FLAG_PLAYER_TARGET) then return false end
	if Duel.GetOperationCount(ev)~=1 then return false end
	local ex,cg,cc,cp,cv=Duel.GetOperationInfo(ev,CATEGORY_DRAW)
	return ex and cv>0
end
function s.activate(e,tp,eg,ep,ev,re,r,rp)
	Duel.ChangeChainOperation(ev,s.repop)
end
function s.repop(e,tp,eg,ep,ev,re,r,rp)
	local ex,cg,cc,cp,cv=Duel.GetOperationInfo(0,CATEGORY_DRAW)
	if cp<2 then
		Duel.DiscardDeck(cp,cv,REASON_EFFECT)
	end
end