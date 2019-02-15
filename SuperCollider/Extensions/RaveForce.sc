RaveForce {

	classvar pattern, key, file, bpm, total_step;
	classvar net, func, task;
	classvar action_space, action, step, action_list;
	classvar length, dur;
	classvar buf_combi;
	classvar synth_to_make, pattern, buf, score, path, outputPath;
	classvar server;

	*start {
		arg pattern, key, file, bpm, total_step;
		dur = 60 / (bpm ? 120) / 4;
		total_step = total_step ? 16;
		action_list = [];
		net = NetAddr("127.0.0.1", 57310);
		outputPath = thisProcess.platform.recordingsDir.replace("\\", "/")
		++ "/" ++ "raveforce.wav";
		path = file ? [
			"/home/qichao/Documents/Coding/_python_code/TR-808Kick01.wav",
			"/home/qichao/Documents/Coding/_python_code/TR-808Snare01.wav",
			"/home/qichao/Documents/Coding/_python_code/TR-808Hat_C01.wav"
		];

		this.bootServer;

		OSCdef(\make, { |msg, time, addr, recvPort|
			// check if the pattern is there
			[msg, time, addr, recvPort].postln;
			task = msg[1];

			score = Pbind(
				\instrument, \default,
				\amp, 0,
				\dur, dur
			).asScore(dur * total_step);

			score.recordNRT(
				outputFilePath: outputPath,
				duration:dur * total_step,
				action:{
					switch (task,
						\drum_loop, {
							this.drumLoopPrepare;
							net.sendMsg(\make, outputPath, "combination", 6);
						},
						\bass_line, {
							this.bassLinePrepare;
							net.sendMsg(\make, outputPath, "continuous", 2);
						}, {
							if (file.isNil.not, {
								buf = [];
								file.size.do{
									buf = buf.add(Buffer.new(server))
								};
							});

							// instead of calling an env written in RaveForce
							// you can call action_space directly from Python

							msg[0] = outputPath; // change /make to outputpath
							msg = msg.add(total_step);
							net.sendMsg(\make, *msg);

					}); // end of switch
				}; // end of action
			);
		}, \make).permanent_(true);

		OSCdef(\reset, { |msg, time, addr, recvPort|
			// reset the pattern step to 0
			[msg, time, addr, recvPort].postln;
			step = 0;
			action_list = [];
			server.freeAll;

			score = Pbind(
				\instrument, \default,
				\amp, 0,
				\dur, dur
			).asScore(dur * total_step);

			score.recordNRT(
				outputFilePath: outputPath,
				duration:dur * total_step,
				action:{ net.sendMsg(\reset, outputPath) }
			);
		}, \reset).permanent_(true);

		OSCdef(\step, { |msg, time, addr, recvPort|
			[msg, time, addr, recvPort].postln;
			step =  msg[1].asInteger + 1;
			2.do{msg.removeAt(0)};
			action = msg;
			action.postln;

			if (step > 0, {

				length = dur * step;

				switch (task,
					\drum_loop, {
						this.drumLoopStep;
					},
					\bass_line, {
						var freq, restP;
						step = msg[1].asInteger + 1;
						freq = msg[2].asFloat.linexp(0, 1, 50, 200);
						restP = (msg[3].asFloat > 0.6).asInt;
						this.bassLineStep(freq, restP);
					}, {
						var n;
						action_list = action_list.add(action);
						n = pattern.patternpairs.atIdentityHash(key);
						pattern.patternpairs[n+1] = action_list;
						score = pattern.asScore(length);
						if (file.isNil.not) {
							buf.do{ |item, n|
								score.add([0, item.allocReadMsg(path[n])]);
							};
						};
					}
				);

				score.recordNRT(
					outputFilePath: outputPath,
					duration:length,
					action:{ net.sendMsg(\step, outputPath) }
				);

			});
		}, \step).permanent_(true);

		OSCdef(\render, { |msg, time, addr, recvPort|
			var loop;
			loop = Buffer.read(server, outputPath, action:{
				{PlayBuf.ar(loop.numChannels, loop, doneAction:2)}.play;
			});
		}, \render).permanent_(true);
	}

	*bootServer {
		server = Server.default;
		server.options.memSize = 1024 * 128;
		server.options.maxNodes = 1024 * 32;
		server.options.numOutputBusChannels = 2;
		server.options.numInputBusChannels = 2;
		server.boot();
	}

	*drumLoopPrepare {

		buf = [];
		3.do{
			buf = buf.add(Buffer.new(server))
		};

		SynthDef(\playBuf, {
			arg buf=99, amp=1; // temp:buf=99 makes empty list silent
			var sig;
			sig = PlayBuf.ar(1, buf, doneAction:2);
			sig = Splay.ar(sig, 2, amp);
			Out.ar(0, sig);
		}).load(server);
	}

	*drumLoopStep {

		buf_combi = buf.powerset.sort({ |a, b| a.size > b.size }).reverse;
		buf_combi.removeAt(7); // remove [bd, sn, hh]
		buf_combi.removeAt(4); // remove [bd, sn] as bd+sn is rare
		action_list = action_list.add(buf_combi[action.asInt]);

		pattern = Pbind(
			\instrument, \playBuf,
			\buf, Pseq(action_list),
			\dur, dur
		);

		score = pattern.asScore(length);
		buf.do{ |item, n|
			score.add([0, item.allocReadMsg(path[n])]);
		};
	}

	// abandoned; for refernece;

	*bassLinePrepare {
		SynthDef(\sawBass, {
			arg freq=440, att=0.01, rel=0.1, cutoff=5000, rq=1, amp=1;
			var sig;
			sig = Saw.ar(freq!2) * Env.perc(att, rel).kr(doneAction: 2);
			sig = RLPF.ar(sig, cutoff, rq, amp);
			Out.ar(0, sig);
		}).load(server);
	}

	*bassLineStep {
		arg freq, rest;
		action_list = action_list.add([freq, rest]);
		[freq, rest].postln;
		pattern = Pbind(
			\instrument, \sawBass,
			\freq, Pseq(action_list.collect(_[0])),
			\amp, Pseq(action_list.collect(_[1])),
			\dur, dur
		);
		score = pattern.asScore(length);
	}
}