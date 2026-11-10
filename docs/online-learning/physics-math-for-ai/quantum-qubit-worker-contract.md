# Qubit calculation and independent-validation Worker contract

2026-11-09. This execution layer serves the four unregistered circuit/channel lessons. Its five request kinds are circuit, channel, environment, instrument and deferred measurement. It does not register lessons or prove production Next.js integration.

## Reader-safe input boundary

`learning-qubit-input.ts` owns and freezes structural inputs using data descriptors. It does not call mathematical code, BigInt or caller getters. Arrays must be dense and all records have exact known fields. Fractions are strings: `"3/5"`, not a rounded number. Each real/imaginary part has two rational coefficients for a+b√2. Numerators and positive denominators allow up to32 digits; exponent notation and decimal-number coefficients are rejected. Equivalent fraction tokens are allowed and retain their identity in the input key; Worker arithmetic reduces them exactly.

The circuit boundary permits1–3 qubits and at most16 gates. Maps support transpose, at most8 weighted2×2 operators or4 matrix-unit images. Instruments have two outcomes and at most8 total weighted operators. An environment parameter must lie in[0,1]. The structural gate does not claim that a density matrix is physical: Workers check that separately. Deliberately nonphysical *maps* remain available for teaching CP/TP failures.

Draft and stored-input JSON are capped at65536 characters. Duplicate decoded keys, fractional/exponent integer tokens, sparse arrays, accessors and unexpected prototypes are rejected. Stored data use `{version:1,value:input}`. These are execution budgets for the finite lab, not limits on the curriculum's theory or research scope.

## Messages and attribution

A calculation request has `{version:1,requestId,input}` with a positive safe-integer ID. The Worker emits a `started` frame and then a success or a failure. A success echoes owned input, its deterministic key, request kind and exact result. Its header alone never authorizes publication.

The unregistered execution contract now also requires `observations: {schemaVersion:1,coordinates,states}` on success. Coordinates mirror the exact report and carry certified dyadic intervals, exact signs and out-of-range nulls. Role-named states include matrix coordinates, physicality and permitted purity/Bloch observations. Instrument roles explicitly distinguish before/after conditional-X feedback; deferred roles distinguish coherent/dephased joint and marginal states. Zero-probability conditional states remain null. These observations must pass independent validation as well as the exact report. See [the display response verification](quantum-qubit-observations-status.md).

A separate validation request carries the original input and calculation response. The trusted validation Worker owns the candidate, checks the ID/kind/input/key and recomputes every observation using the independent mathematical validators. Only then does it return a `validated` receipt containing the checked response. A reader must consume that receipt's response through the controller, rather than publish the raw calculation result or trust a Boolean sent by the calculation Worker.

`ready` means the calculation was verified. It does **not** mean an inspected map is CPTP; that conclusion comes from the result's separate physicality properties. Invalid input states, mathematically correct counterexamples and corrupted results are distinct cases.

## Ownership, responsiveness and lifecycle

Response snapshots permit ordinary finite JSON data only, with limits of100000 nodes, depth32,256 array entries,48 record fields,65536 characters per string and2000000 total string characters. Cycles and accessors are rejected; shared subtrees are owned once. Worker copies are synchronous off the UI thread. Controller copies yield after1024 nodes or about4ms and check whether the request is still active.

Each run owns one calculation Worker and one trusted independent-validation Worker. Replacement, cancellation, disposal or failure detach handlers and call `terminate()` on owned ports. Active-token checks prevent captured callbacks and late receipts from publishing under a new input. The default total deadline is30000ms, configurable from10 to120000ms. It is a browser timer, not a guarantee about background-tab scheduling.

The controller distinguishes invalid input, unavailable Workers, Worker failure, invalid response, resource budget and timeout. Retry calls `run` again with owned input and a new request ID. There is no synchronous mathematical fallback in the reader. Factories use module Workers with `new URL(..., import.meta.url)` for the repository's later Next.js build.

This uses the [HTML Standard's dedicated Worker messaging and termination model](https://html.spec.whatwg.org/multipage/workers.html). The ownership checks, mathematical admission and cancellation tokens are application-level requirements in addition to browser message cloning.

## Verification boundary

The controller bundle excludes mathematical engines. The validation bundle excludes product circuit/channel engines and imports only common arithmetic, decoding and independent validators. Node tests cover every prior595 independent preparation, valid-format scalar forgeries, attribution, decoder failures and lifecycle races. Browser checks use actual module Workers from locally bundled Blob URLs, compare all595 results and validation receipts with independent reference hashes, and exercise cancellation, replacement, failure, timeout and typing during pending work.

Blob-worker verification does not prove Next's emitted asset URLs, actual lesson pages, storage recovery, accessibility or translated UI. Those remain gates before registration under **Book → Interactive Book**. Actual hidden-tab automation is also not claimed.
