"""Independent time quadrature, scalar flow and finite-population Jacobian checks."""
from fractions import Fraction as F
from itertools import combinations_with_replacement
import math,json
import numpy as np

def simpson(f,a,b,n=4096):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+sum((4 if i%2 else 2)*f(a+i*h)for i in range(1,n)))

def flow(delta,coupling,initial,times):
    state=initial;prior=0;rows=[];f=lambda x:delta-coupling*math.sin(x)
    for target in times:
        steps=max(1,math.ceil((target-prior)*max(1,abs(delta))/.002));h=(target-prior)/steps
        for _ in range(steps):
            k1=f(state);k2=f(state+h*k1/2);k3=f(state+h*k2/2);k4=f(state+h*k3)
            state+=h*(k1+2*k2+2*k3+k4)/6
        rows.append(state);prior=target
    return rows

def jacobian(points,k):
    n=len(points)
    return [[k/n*(points[i][0]*points[j][0]+points[i][1]*points[j][1])if i!=j else -sum(k/n*(points[i][0]*points[h][0]+points[i][1]*points[h][1])for h in range(n)if h!=i)for j in range(n)]for i in range(n)]

def verify():
    occupation=0;reflection=0;sensitivities=0;flows=0;observations=0;max_quad=0;max_turn=0;max_clock=0
    for delta in [.5,1.,2.,3.]:
        for ratio in [0.,.2,.6,.8,.96]:
            k=delta*ratio;q=math.sqrt(delta*delta-k*k);period=2*math.pi/q;p=lambda x:1/(period*(delta-k*math.sin(x)))
            target_sine=k/(delta+q);half=.5+math.atan(k/q)/math.pi
            checks=[(simpson(p,0,2*math.pi),1),(simpson(lambda x:p(x)*math.sin(x),0,2*math.pi),target_sine),(simpson(lambda x:p(x)*math.cos(x),0,2*math.pi),0),(simpson(lambda x:p(x)*(1+math.cos(x))/2,0,2*math.pi),.5),(simpson(p,0,math.pi),half)]
            for actual,target in checks:
                error=abs(actual-target);max_quad=max(max_quad,error);assert error<2e-11
            if ratio:
                eps=delta*1e-6
                numeric=(simpson(lambda x:1/(delta-(k+eps)*math.sin(x)),0,2*math.pi)-simpson(lambda x:1/(delta-(k-eps)*math.sin(x)),0,2*math.pi))/(2*eps)
                analytic=2*math.pi*k/q**3;assert abs(numeric/analytic-1)<2e-8;sensitivities+=1
            occupation+=1
            for x in np.linspace(-math.pi,math.pi,41):
                assert abs(p(-x)-q/(2*math.pi*abs(-delta-k*math.sin(x))))<1e-12;reflection+=1
            for sign in [-1,1]:
                for initial in [.3,2.1]:
                    times=[period*.125,period*.7,period];rows=flow(sign*delta,k,initial,times)
                    def clock(x):
                        turns=math.floor(x/(2*math.pi));wrapped=x-turns*2*math.pi
                        return turns*2*math.pi+q*simpson(lambda u:1/(delta-k*math.sin(u)),0,wrapped)
                    start=clock(sign*initial)
                    for t,state in zip(times,rows):
                        assert abs((state-initial)/t-sign*q)<2*math.pi/t+1e-10
                        error=abs(clock(sign*state)-start-q*t);max_clock=max(max_clock,error);assert error<2e-8;observations+=1
                    error=abs(rows[-1]-initial-sign*2*math.pi);max_turn=max(max_turn,error);assert error<2e-8;flows+=1
    directions=[(F(1),F(0)),(F(3,5),F(4,5)),(F(0),F(1)),(F(-3,5),F(4,5)),(F(-5,13),F(12,13)),(F(5,13),F(12,13))]
    balanced=0;finite_difference=0;max_spectrum=0;max_diff=0
    for pairs in range(1,5):
        for chosen in combinations_with_replacement(directions,pairs):
            points=[p for x,y in chosen for p in [(x,y),(-x,-y)]];n=len(points)
            z2=(sum(x*x-y*y for x,y in points)/n,sum(2*x*y for x,y in points)/n);r2sq=sum(x*x for x in z2)
            for k in [F(0),F(1),F(3)]:
                J=jacobian(points,k)
                assert all(sum(row)==0 for row in J)
                assert all(J[i][j]==k/n*sum(points[i][h]*points[j][h]for h in range(2))for i in range(n)for j in range(n))
                assert sum(J[i][i]for i in range(n))==k
                assert sum(J[i][j]*J[j][i]for i in range(n)for j in range(n))==k*k*(1+r2sq)/2
                actual=np.linalg.eigvalsh(np.array(J,dtype=float));r2=math.sqrt(float(r2sq));expected=sorted([float(k)*(1-r2)/2,float(k)*(1+r2)/2]+[0.]*(n-2));error=float(np.max(np.abs(actual-expected)));max_spectrum=max(max_spectrum,error);assert error<2e-12
                if balanced%7==0:
                    angles=[math.atan2(float(y),float(x))for x,y in points];eps=2e-6
                    def rhs(a):return [float(k)/n*sum(math.sin(a[j]-a[i])for j in range(n))for i in range(n)]
                    for j in range(n):
                        up=angles[:];down=angles[:];up[j]+=eps;down[j]-=eps;fu=rhs(up);fd=rhs(down)
                        for i in range(n):error=abs((fu[i]-fd[i])/(2*eps)-float(J[i][j]));max_diff=max(max_diff,error);assert error<2e-9
                    finite_difference+=1
                balanced+=1
    aligned=0;cohesive=0
    for n in range(1,17):
        for k in [F(0),F(1),F(3)]:
            J=jacobian([(F(1),F(0))]*n,k);actual=np.linalg.eigvalsh(np.array(J,dtype=float));assert np.max(np.abs(actual-sorted([0.]+[-float(k)]*(n-1))))<2e-12;aligned+=1
    for n in range(2,9):
        points=[]
        for i in range(n):
            a=F(i%4,10);points.append(((1-a*a)/(1+a*a),2*a/(1+a*a)))
        for k in [F(1),F(3)]:
            J=jacobian(points,k);omega=[-k/n*sum(points[j][1]*points[i][0]-points[j][0]*points[i][1]for j in range(n))for i in range(n)];assert sum(omega)==0
            assert all(omega[i]+k/n*sum(points[j][1]*points[i][0]-points[j][0]*points[i][1]for j in range(n))==0 for i in range(n))
            eta=[F(i)-F(n-1,2)for i in range(n)];lhs=sum(eta[i]*J[i][j]*eta[j]for i in range(n)for j in range(n));rhs=-k/n*sum((points[i][0]*points[j][0]+points[i][1]*points[j][1])*(eta[i]-eta[j])**2 for i in range(n)for j in range(i+1,n));assert lhs==rhs
            lower=min(sum(points[i][h]*points[j][h]for h in range(2))for i in range(n)for j in range(n));assert lower>0;assert lhs<=-k*lower*sum(x*x for x in eta)
            spectrum=np.linalg.eigvalsh(np.array(J,dtype=float));assert abs(spectrum[-1])<2e-12 and spectrum[-2]<=-float(k*lower)+2e-12;cohesive+=1
    return {'status':'passed','occupationPreparations':occupation,'reflectionPoints':reflection,'periodSensitivityPreparations':sensitivities,'scalarFlowPreparations':flows,'finiteWindowObservations':observations,'exactBalancedPreparations':balanced,'jacobianDifferencePreparations':finite_difference,'alignedPreparations':aligned,'exactCohesiveLockedPreparations':cohesive,'maxOccupationQuadratureError':max_quad,'maxFullTurnPhaseError':max_turn,'maxUniformClockError':max_clock,'maxSpectrumError':max_spectrum,'maxJacobianDifferenceError':max_diff,'numericalReference':'Composite Simpson, independent scalar RK4 and full symmetric eigensolver; floating-point comparisons, not directed error certificates'}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
