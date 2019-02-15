(
var pattern;

SynthDef(\sin, { |freq, att=0.01, rel=0.49|
	var sig, freq_range;
	freq_range = freq.linexp(0, 1, 20, 22000);
	sig = SinOsc.ar(freq_range!2);
	sig = sig * Env.perc(att, rel).kr(2);
	Out.ar(0, sig);
}).load;

pattern = Pbind(
	\instrument, \sin,
	\freq, 2000,
	\dur, 0.5,
);

RaveForce.start(pattern, key: \freq, bpm:30, total_step:1);

)