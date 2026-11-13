"""Independent microscopic-path filtering and augmented linear-flow verification."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import importlib.util,sys,json,math
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('response_reference',Path(__file__).with_name('verify-response-research.py'));reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
T=[[F(1,2),F(1,4),F(1,4)],[F(1,4),F(0),F(3,4)],[F(1,4),F(3,4),F(0)]]
def multiply(T,beta):return [sum(T[i][j]*beta[j]for j in range(3))for i in range(3)]
def filter_history(T,initial,history):
    beta=initial[:];mass=F(1)
    for k,label in enumerate(history):
        if k:beta=multiply(T,beta)
        masked=[v if(i<2)==(label=='A')else F(0)for i,v in enumerate(beta)];normalizer=sum(masked)
        if not normalizer:return None,F(0)
        beta=[v/normalizer for v in masked];mass*=normalizer
    return beta,mass

def verify():
    matrices=[T,[[F(i==j)for j in range(3)]for i in range(3)],[[F(1,3)]*3 for _ in range(3)],[[F(i==(j+1)%3)for j in range(3)]for i in range(3)]]
    initials=[[F(1,3)]*3,[F(1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(1)],[F(1,2),F(1,2),F(0)]]
    paths=histories=impossible=futures=0
    for transition,initial in product(matrices,initials):
        for length in range(1,7):
            brute={}
            for state_path in product(range(3),repeat=length):
                weight=initial[state_path[0]]
                for i,j in zip(state_path,state_path[1:]):weight*=transition[j][i]
                paths+=1
                if not weight:continue
                labels=''.join('A'if i<2 else'B'for i in state_path);brute.setdefault(labels,[F(0)]*3)[state_path[-1]]+=weight
            assert sum(sum(row)for row in brute.values())==1
            for label_path in product('AB',repeat=length):
                history=''.join(label_path);beta,mass=filter_history(transition,initial,history);truth=brute.get(history,[F(0)]*3)
                if not sum(truth):assert beta is None and mass==0;impossible+=1;continue
                assert mass==sum(truth)and beta==[v/mass for v in truth]and sum(beta)==1;histories+=1
                if length<=3:
                    predicted=beta[:]
                    for horizon in range(1,4):
                        predicted=multiply(transition,predicted);future=[F(0)]*3
                        for start in range(3):
                            for tail in product(range(3),repeat=horizon):
                                weight=truth[start]/mass;prior=start
                                for state in tail:weight*=transition[state][prior];prior=state
                                future[tail[-1]]+=weight
                        assert predicted==future;futures+=1
    # Independent three-label probabilities and ideal prediction losses.
    triple={}
    for i,j,k in product(range(3),repeat=3):
        labels=tuple('A'if n<2 else'B'for n in [i,j,k]);triple[labels]=triple.get(labels,F(0))+F(1,3)*T[j][i]*T[k][j]
    mid={b:sum(v for(a,x,c),v in triple.items()if x==b)for b in 'AB'}
    first={(a,b):sum(v for(x,y,c),v in triple.items()if(x,y)==(a,b))for a,b in product('AB',repeat=2)}
    last={(b,c):sum(v for(a,x,y),v in triple.items()if(x,y)==(b,c))for b,c in product('AB',repeat=2)}
    info=coarse_loss=history_loss=0.;mse_gain=F(0)
    for(a,b,c),v in triple.items():
        if not v:continue
        pbase=last[b,c]/mid[b];pfull=v/first[a,b];info+=float(v)*math.log(float(pfull/pbase));coarse_loss-=float(v)*math.log(float(pbase));history_loss-=float(v)*math.log(float(pfull))
    for a,b in product('AB',repeat=2):
        if first[a,b]:mse_gain+=first[a,b]*(triple.get((a,b,'B'),F(0))/first[a,b]-last[b,'B']/mid[b])**2
    expected=2/3*(5/8*math.log(5/4)+3/8*math.log(3/4));assert abs(info-expected)<1e-15 and abs(coarse_loss-history_loss-info)<1e-15 and mse_gain==F(1,96)
    assert filter_history(T,[F(1,3)]*3,'AAA')[0][0]==F(7,10)and filter_history(T,[F(1,3)]*3,'BAA')[0][0]==F(5,6)
    linear=observations=laplaces=0;max_flow=max_identity=max_fast=0.;scaled_error=0
    for a,d,b,c in product([F(1),F(2)],[F(1),F(2)],[F(-1),F(0),F(1)],[F(-1),F(0),F(1)]):
        assert a*d-b*c>=0
        for eps,(x0,y0) in product([F(1,16),F(1,32),F(1,64)],[(F(1),F(0)),(F(1),F(1)),(F(0),F(1)),(F(-1),F(1))]):
            mu=-a+b*c/d;w0=y0-c*x0/d;rate=mu/(1+eps*b*c/d**2);slip=eps*b*w0/d
            for s in [F(1),F(2),F(3)]:
                actual=reference.solve([[s+a,-b],[-c/eps,s+d/eps]],[x0,y0])[0];target=((eps*s+d)*x0+eps*b*y0)/((s+a)*(eps*s+d)-b*c);assert actual==target;laplaces+=1
            af,bf,cf,df,ef=map(float,[a,b,c,d,eps]);disc=math.sqrt((df-ef*af)**2+4*ef*bf*cf);slow=-2*float(a*d-b*c)/(df+ef*af+disc);fast=-(df+ef*af)/ef-slow;aslow=(float(-a*x0+b*y0)-fast*float(x0))/(slow-fast);afast=float(x0)-aslow
            assert fast<=-df/(2*ef)
            M=[[-af,bf,0.],[cf/ef,-df/ef,0.],[-af,bf,-df/ef]]
            for t in sorted(set([0.,ef/df*.1,ef/df,4*ef/df,.1,.5,1.])):
                expm=reference.matrix_exp(M,t);state=[sum(row[j]*float([x0,y0,F(0)][j])for j in range(3))for row in expm];x,y,z=state;exact=aslow*math.exp(slow*t)+afast*math.exp(fast*t);flow_error=abs(exact-x);max_flow=max(max_flow,flow_error);assert flow_error<2e-11
                residual=(-af*x+bf*y)-float(mu)*x-bf*float(w0)*math.exp(-df*t/ef)+float(b*c/d)*z;max_identity=max(max_identity,abs(residual));assert abs(residual)<2e-11
                difference=abs(math.exp(fast*t)-math.exp(-df*t/ef));bound=2*ef/(math.e*df)*abs(fast+df/ef);assert difference<=bound+2e-14;max_fast=max(max_fast,difference)
                comp=float(x0+slip)*math.exp(float(rate)*t)-float(slip)*math.exp(-df*t/ef);scaled_error=max(scaled_error,abs(x-comp)/ef**2);observations+=1
            linear+=1
    return {'status':'passed','filterTransitionFamilies':4,'filterInitialPreparations':5,'enumeratedMicroPaths':paths,'positiveHistoryPreparations':histories,'impossibleHistoryPreparations':impossible,'futureConditionalPreparations':futures,'stationaryConditionalInformation':info,'squaredLossImprovement':str(mse_gain),'linearModePreparations':linear,'augmentedFlowObservations':observations,'exactLaplacePreparations':laplaces,'maxTwoModeFlowDifference':max_flow,'maxIntegrationByPartsResidual':max_identity,'maxFastExponentialDifference':max_fast,'maxSampledCompositeErrorDividedByEpsilonSquared':scaled_error,'numericalReference':'Exact Fraction path enumeration and Laplace solves; independent augmented matrix exponential uses floating-point Taylor scaling/squaring, not a directed certificate'}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
