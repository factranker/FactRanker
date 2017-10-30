import extractors.tokenizer as tokenizer

#import tokenizer
from collections import Counter
import re



feel = [ "brush*","caress*" ,"cold*" ,"cool*" ,"concert*" ,"drily" ,"dry*" ,"edge" ,"harmon*" ,"hear" ,"feel" ,"feeling*" ,"feels" ,"felt" ,"inaudibl*" ,"fire" ,"flexib*" ,"fragil*" ,"freez*" ,"froze*" ,"fuzz*" ,"grab*" ,"grip" ,"gripp*" ,"grips" ,"hair*" ,"hand" ,"handful*" ,"ringing" ,"hard" ,"harde*" ,"heavie*" ,"say*" ,"hot" ,"pink*" ,"leather*" ,"limp*" ,"loose*" ,"press" ,"pressed" ,"presser*" ,"presses" ,"speech*" ,"round*" ,"rub" ,"voic*" ,"sees" ,"rubs" ,"sand" ,"yelling" ,"sandy" ,"scratch*" ,"sharp*" ,"silk*" ,"skin" , "stroki*" ,"thick*" ,"thin" ,"viewer*" ,"weighted" ,"weightless*" ,"weightlift*" ,"weights" ,"wetly" ]  

percept =  ["acid*" ,"acrid*" ,"aroma*" ,"audibl*" ,"if" ,"beaut*" ,"bitter*" ,"black" ,"rather" ,"really" ,"blacks" ,"blind*" ,"blond*" ,"blue*" ,"vs" ,"bright*" ,"brown*" ,"brush*" ,"butter*" ,"candle*" ,"caramel*" ,"caress*" ,"chocolate*" ,"choir*" ,"circle" ,"citrus*" ,"click*" ,"cold*" ,"color*" ,"colour*" ,"column*" ,"cool*" ,"deaf*" ,"delectabl*" ,"delicious*" ,"deoder*" ,"drie*" ,"drily" ,"drool*" ,"ear" ,"ears" ,"fenc*" ,"edging" ,"experienc*" ,"eying" ,"feel" ,"feeling*" ,"feels" ,"felt" ,"flavour*" ,"flexib*" ,"fragil*" ,"gaz*" ,"glow*" ,"grab*" ,"gray*" ,"green*" ,"grey*" ,"gripp*" ,"grips" ,"hair*" ,"hand" ,"hands" ,"harde*" ,"harmon*" ,"hear" ,"heard" ,"heavy*" ,"honey" ,"hott*" ,"hush*" ,"image*" ,"inaudibl*" ,"limp*" ,"listen" ,"listened" ,"listener*" ,"listening" ,"listens" ,"lit" ,"look" ,"looker*" ,"looks" ,"loose*" ,"loud*" ,"mint*" ,"musi*" ,"noise" ,"noises" ,"nose*" ,"nostril*" ,"odor*" ,"oil*" ,"orange*" ,"palatabl*" ,"perfum*" ,"picture" ,"pink*" ,"press" ,"presser*" ,"presses" ,"pungen*" ] 

certain = [ "absolute" ,"absolutely" ,"accura*" ,"all" ,"any" ,"always" ,"apparent" ,"assur*" ,"anything" ,"anytime" ,"clear" ,"clearly" ,"commit" ,"commitment*" ,"appearing" ,"committ*" ,"complete" ,"completed" ,"completely" ,"completes" ,"confidence" ,"confident" ,"confidently" ,"correct*" ,"defined" ,"definite" ,"definitely" ,"definitive*" ,"depend" ,"distinct*" ,"entire*" ,"essential" ,"disorient*" ,"every" ,"needs" ,"everything*" ,"evident*" ,"exact*" ,"explicit*" ,"extremely" ,"fact" ,"facts" ,"guessing" ,"forever" ,"frankly" ,"hazie*" ,"how" ,"fundamentally" ,"fundamentals" ,"hoped" ,"implicit*" ,"indeed" ,"inevitab*" ,"infallib*" ,"invariab*" ,"hypothes*" ,"hypothetic*" ,"if" ,"incomplet*" ,"indecis*" ,"mustve" ,"must've" ,"necessar*" ,"(of)" ,"wanting" ,"kindof" ,"likel*" ,"lot" ,"proof" ,"pure*" ,"sure*" ,"total" ,"wouldve" ,"true" ,"truest" ,"truth*" ,"unambigu*" ,"undeniab*" ,"undoubt*" ,"wholly" ,"mightve" ,"nearly" ,"occasional*" ,"option" ,"possib*" ,"practically" ,"pretty" ,"probable" ,"seeming*" ,"shaki*" ,"something*" ,"somewhat" ,"sorts" ,"tempora*" ,"unluck*" ,"unsettl*" ,"unsure*"] 

time = ["abrupt*" ,"after" ,"afterlife*" ,"aftermath*" ,"mile*" ,"afterthought*" ,"afterward*" ,"again" ,"neared" ,"nearer" ,"nearest" ,"aging" ,"ago" ,"ahead" ,"bending" ,"always" ,"ancient*" ,"annual*" ,"anymore" ,"anytime" ,"april" ,"august" ,"autumn" ,"awhile" ,"back" ,"before" ,"began" ,"begin" ,"capacit*" ,"begins" ,"begun" ,"biannu*" ,"point" ,"birth*" ,"closed" ,"born" ,"busy" ,"bye" ,"roomate*" ,"roomed" ,"centur*" ,"childhood" ,"roommate*" ,"rooms" ,"common" ,"segment" ,"deliver*" ,"continu*" ,"current*" ,"short*" ,"dail*" ,"date*" ,"day*" ,"decade*" ,"decay*" ,"sky*" ,"small*" ,"somewhere" ,"south*" ,"space" ,"early" ,"end" ,"ended" ,"span" ,"farther" ,"farthest" ,"straight" ,"floor*" ,"evening*" ,"surfac*" ,"eventually" ,"ever" ,"everyday" ,"ginormous" ,"fading*" ,"fast" ,"hall" ,"fastest" ,"february" ,"final" ,"finally" ,"huge*" ,"first" ,"top" ,"firsts" ,"followup*" ,"forever" ,"former*" ,"undersid*" ,"growing" ,"up" ,"frequenting" ,"frequently" ,"uppermost" ,"friday*" ,"futur*" ,"verg*" ,"vertical*" ,"wall" ,"walling" ,"walls" ,"west*" ,"where" ,"immediately" ,"immediateness" ,"immortal*" ,"inciden*" ,"infinit*" ,"initial*" ,"initiat*" ,"interval*" ,"july" ,"last*" ,"late" ,"lately" ] 



names = ["feel", "percept", "certain", "time"]

liwc_cats = [feel, percept, certain, time]

def feature_names():
    return ["liwccat_"+ i for i in names]

def feature_name_type():
    return [("liwccat_"+ i, 'NUMERIC') for i in names]


def features(text):
	parsed = tokenizer.parse(text)
	c = Counter()
	for sentence in parsed:
		tokens = sentence['tokens']
		for tokeninfo in tokens:
			word = tokeninfo['word']
			for name,cat in zip(names, liwc_cats):
				for reg in cat:
					if(re.match(reg, word)):
						c[name] += 1
						# print(word)


	feature = [0]*len(names)

	for i,name in enumerate(names):
		feature[i] = c[name]
		# print(name,c[name])

	return feature

print("Testing: LIWC_Cat")
print(features("Twenty-nine million people have no health insurance today in America"))
