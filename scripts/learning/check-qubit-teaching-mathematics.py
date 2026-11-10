"""Exact audit of all13 guided explanations against independent reference data."""
import hashlib
import json
from pathlib import Path
import sympy as S

root=Path('.runtime/learning/quantum-qubit-authoring');root.mkdir(parents=True,exist_ok=True)
files=['scripts/fixtures/qubit-circuit-response-independent.json','scripts/fixtures/qubit-channel-independent.json']
circuit=json.loads(Path(files[0]).read_text());channel=json.loads(Path(files[1]).read_text())
maps={'circuit':{c['name']:c['expected'] for c in circuit['cases']}}
for key,kind in [('channelCases','channel'),('environmentCases','environment'),('instrumentCases','instrument'),('deferredCases','deferred')]:maps[kind]={c['name']:c['expected'] for c in channel[key]}
def scalar(w):return S.Rational(w['real'][0])+S.Rational(w['real'][1])*S.sqrt(2)+S.I*(S.Rational(w['imaginary'][0])+S.Rational(w['imaginary'][1])*S.sqrt(2))
def mat(w):return S.Matrix([[scalar(v) for v in row] for row in w])
def purity(a):return S.simplify((a*a).trace())
def get(kind,name):return maps[kind][name]
def final(name):return get('circuit',name)['prefixes'][-1]
X=S.Matrix([[0,1],[1,0]]);Y=S.Matrix([[0,-S.I],[S.I,0]]);I=S.eye(2);P0=S.diag(1,0)
proofs={}
assert [scalar(v) for v in final('HZH')['probabilities']]==[0,1]
assert [scalar(v) for v in final('ZHH')['probabilities']]==[1,0]
proofs['circuit-order']=dict(verified=True,HZH='|1>',ZHH='|0>')
assert S.simplify(scalar(final('HTH')['probabilities'][1])-(2-S.sqrt(2))/4)==0
for step in [1,2]:assert [scalar(v) for v in get('circuit','HTH')['prefixes'][step]['probabilities']]==[S.Rational(1,2)]*2
g=get('circuit','global-T');assert g['samePhysicalStateAsInput'] and not g['sameAmplitudesAsInput'] and scalar(g['fidelityWithInput'])==1
proofs['circuit-phase']=dict(verified=True,finalP1='(2-sqrt(2))/4',globalPhaseSquaredFidelity='1')
bell=final('bell-control-0');ghz=final('GHZ')
assert scalar(bell['singleQubits'][0]['purity'])==scalar(ghz['singleQubits'][0]['purity'])==S.Rational(1,2)
assert scalar(bell['pauliPairs'][0]['expectations'][0][0])==1 and scalar(ghz['pauliPairs'][0]['expectations'][0][0])==0
rho=mat(ghz['density']);mixture=S.diag(S.Rational(1,2),0,0,0,0,0,0,S.Rational(1,2));XXX=S.kronecker_product(X,X,X)
for forgotten in range(3):
 a=S.tensorcontraction(S.MutableDenseNDimArray(list(rho),[2]*6),(forgotten,forgotten+3));b=S.tensorcontraction(S.MutableDenseNDimArray(list(mixture),[2]*6),(forgotten,forgotten+3));assert a==b
assert (rho*XXX).trace()==1 and (mixture*XXX).trace()==0
proofs['circuit-joint']=dict(verified=True,BellXX='1',GHZPairXX='0',GHZXXX='1',classicalMixtureXXX='0')
damp=get('channel','damping-1/2/plus-i');out=mat(damp['output'])
assert out==S.Matrix([[S.Rational(41,50),-3*S.I/10],[3*S.I/10,S.Rational(9,50)]]) and purity(out)==S.Rational(553,625)
assert mat(damp['unitalImage'])==S.diag(S.Rational(41,25),S.Rational(9,25)) and mat(damp['dualIdentity'])==I
proofs['channel-damping']=dict(verified=True,outputPurity='553/625',unital=False,tracePreserving=True)
trans=get('channel','transpose/plus-i');anti=S.Matrix([0,1,-1,0])/S.sqrt(2)
assert (anti.H*mat(trans['bellOutput'])*anti)[0]==-S.Rational(1,2) and trans['outputIsDensity'] and not trans['properties']['completelyPositive']
proofs['channel-reference']=dict(verified=True,negativeBellExpectation='-1/2')
assert mat(get('channel','dephasing-1/2/plus-i')['output'])==I/2
assert (mat(get('channel','dephasing-1/plus-i')['output'])*Y).trace()==-1
assert mat(get('channel','postselect-zero/one')['output'])==S.zeros(2) and scalar(get('channel','trace-growing/plus-i')['outputTrace'])==2
proofs['channel-parameters']=dict(verified=True,pHalfPurity='1/2',pOneY='-1',impossibleSelectionTrace='0',growingTrace='2')
env=get('environment','1/2/plus-i')
assert purity(mat(env['joint']))==1 and purity(mat(env['system']))==purity(mat(env['environment']))==S.Rational(553,625)
assert mat(get('environment','1/plus-i')['system'])==P0
assert mat(get('environment','1/plus-i')['environment'])==S.Matrix([[S.Rational(1,2),-S.I/2],[S.I/2,S.Rational(1,2)]])
proofs['environment-transfer']=dict(verified=True,partialLocalPurity='553/625',jointPurity='1',completeTransferEnvironment='input')
assert purity(mat(get('environment','1/2/one')['system']))==S.Rational(337,625)
assert purity(mat(get('environment','1/2/zero')['system']))==1
mixed=get('environment','1/2/mixed');actual=[purity(mat(mixed[k])) for k in ['joint','system','environment']]
assert actual==[S.Rational(11,18),S.Rational(1011,1250),S.Rational(3937,5625)]
proofs['environment-inputs']=dict(verified=True,excitedLocalPurity='337/625',mixedPurities=list(map(str,actual)))
a=get('instrument','damping-record/1/2/zero/False');b=get('instrument','damping-mixed-record/1/2/zero/False')
assert [scalar(x['probability']) for x in a['branches']]==[1,0]
assert [scalar(x['probability']) for x in b['branches']]==[S.Rational(1,2)]*2
assert mat(a['forgotten'])==mat(b['forgotten'])==P0
proofs['environment-basis']=dict(verified=True,computationalRecord=['1','0'],plusMinusRecord=['1/2','1/2'])
assert get('instrument','projective-z/1/2/zero/False')['branches'][1]['conditional'] is None
coarse=get('instrument','coarse-pauli-record/1/2/plus-i/False');conditional=mat(coarse['branches'][0]['conditional'])
assert scalar(coarse['branches'][0]['probability'])==S.Rational(3,4) and (conditional*Y).trace()==S.Rational(1,3) and purity(conditional)==S.Rational(5,9)
assert mat(coarse['forgotten'])==I/2
proofs['instrument-branches']=dict(verified=True,coarseProbability='3/4',conditionalY='1/3',conditionalPurity='5/9',zeroConditional=None)
projective=get('instrument','projective-z/1/2/plus/False');reset=get('instrument','reset-z/1/2/plus/False');corrected=get('instrument','projective-z/1/2/plus/True')
assert [x['effect'] for x in projective['branches']]==[x['effect'] for x in reset['branches']]
assert mat(projective['forgotten'])==I/2 and mat(reset['forgotten'])==mat(corrected['feedforward'])==P0
proofs['instrument-updates']=dict(verified=True,effectsAgree=True,projectiveForgotten='I/2',resetAndCorrected='|0><0|')
proofs['instrument-record-basis']=dict(verified=True,forgottenChannelsAgree=True,recordsDiffer=True)
deferred=get('deferred','plus-i');assert mat(deferred['system'])==mat(get('instrument','projective-z/1/2/plus-i/True')['feedforward'])==P0
assert (mat(deferred['record'])*Y).trace()==1
dephased=mat(deferred['dephased']);record=S.Matrix(2,2,lambda i,j:sum(dephased[2*k+i,2*k+j] for k in range(2)))
assert (record*Y).trace()==0 and record==I/2
proofs['instrument-deferral']=dict(verified=True,coherentRecordY='1',dephasedRecordY='0',system='|0><0|')
assert len(proofs)==13
report=dict(guidedClaimsVerified=13,proofs=proofs,sourceHashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
(root/'mathematical-authoring-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(guidedClaimsVerified=13,mixedPurities=list(map(str,actual)))))
