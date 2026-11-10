"""Account for successful request envelopes andvalidation receipts on top of
the independently derived producer string/structure caps. No limits increase.
Run verify-real-time-producer-bounds.py first when producer layouts change.
"""
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parents[2]
producer_file = base / '.runtime/learning/real-time-phase-numerics/producer-bounds.json'
producer = json.loads(producer_file.read_text())
# Input schemas have<=16fields;fraction text has<=26characters including
# slash andsign. Allkeys,enum labels andJSONsyntax fit in2048characters.
input_value_chars = 16 * 26 + 64
input_key_chars = 2048
assert input_value_chars + 16 * 32 + 16 * 8 < input_key_chars
# A receipt contains its inputKey andthe envelope's inputKey,owned input,
# plus fixed labels. Producer root/exact input copies are already inits cap.
header_value_chars = input_value_chars + 2 * input_key_chars + 256
assert header_value_chars < 8192
handoff_value_chars = 2 * input_value_chars + input_key_chars + 256
assert handoff_value_chars <= header_value_chars
reserve = 16384
caps = {mode: producer[mode + 'StringCharactersUpper'] + reserve
        for mode in ('chain', 'packet', 'fold')}
nodes = producer['nodesUpper'] + 512
depth = producer['depthUpper'] + 2
single = max(producer['singleStringCharactersUpper'], input_key_chars)
assert max(caps.values()) < 2000000
assert single < 65536 and nodes < 100000 and depth < 32
assert producer['arrayLengthUpper'] <= 256
files = ['src/lib/learning-real-time-phase-validation.ts', 'src/lib/learning-real-time-phase-protocol.ts']
result = dict(scope='whole successful producer envelope andvalidation receipt',
              inputValueCharactersUpper=input_value_chars, inputKeyCharactersUpper=input_key_chars,
              headerValueCharactersUpper=header_value_chars, reservedHeaderCharacters=reserve,
              validationHandoffHeaderCharactersUpper=handoff_value_chars,
              totalStringCharactersUpperByMode=caps, singleStringCharactersUpper=single,
              nodesUpper=nodes, depthUpper=depth, arrayLengthUpper=producer['arrayLengthUpper'],
              existingLimitsUnchanged=True, workerRuntimeDeadlineProved=False,
              producerBoundSha256=hashlib.sha256(producer_file.read_bytes()).hexdigest(),
              sourceSha256={p: hashlib.sha256((base / p).read_bytes()).hexdigest() for p in files})
root = base / '.runtime/learning/real-time-whole-response'
root.mkdir(parents=True, exist_ok=True)
(root / 'wire-bounds.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
