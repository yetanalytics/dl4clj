(ns ^{:doc ""}
  dl4clj.examples.rnn.tools
  (:require [nd4clj.linalg.dataset.api.iterator.data-set-iterator :refer (input-columns total-outcomes)]
            [nd4clj.linalg.factory.nd4j :refer (zeros)]
            [nd4clj.linalg.api.ndarray.indarray :refer (put-scalar get-double tensor-along-dimension)]
            [dl4clj.nn.multilayer.multi-layer-network :refer (rnn-clear-previous-state rnn-time-step)])
  (:import [java.util Random]))

(defn- sample-from-distribution 
  "Sample from a probability distribution over discrete classes given as a vector of probabilities
  summing to 1.0."
  ([distribution] 
   (sample-from-distribution distribution (Random.)))
  ([distribution ^Random rng] 
   (let [d (.nextDouble rng)]
     (loop [i 0
            sum (nth distribution 0)]
       (cond (<= d sum) i
             (< i (count distribution)) (recur (inc i) (+ sum (nth distribution (inc i))))
             :else (throw (IllegalArgumentException. (str "Distribution is invalid? d= " d ", sum=" sum))))))))

(defn sample-characters-from-network [initialization  net iter rng characters-to-sample num-samples]
  ;; Set up initialization. If no initialization: use a random character
  (let [;; set up initialization. If no initialization: use a random character.
        initialization (or initialization (str (rand-nth (seq (:valid-characters iter)))))
        char-to-idx-map (:char-to-idx-map iter)
        idx-to-char-map (zipmap (vals char-to-idx-map) (keys char-to-idx-map))
        initialization-input (zeros [num-samples (input-columns iter) (count initialization)])]
    ;; create input for initialization
    (dotimes [i (count initialization)]
      (let [idx (char-to-idx-map (nth initialization i))]
        (dotimes [j num-samples]
          (put-scalar initialization-input [j idx i] 1.0))))
    ;; Sample from network (and feed samples back into input) one character at a time (for all
    ;; samples). Sampling is done in parallel here
    (rnn-clear-previous-state net)
    (let [output (rnn-time-step net initialization-input)
          sb (for [i (range num-samples)] (StringBuilder.))]
      (loop [output (tensor-along-dimension output (- (.size output 2) 1) [1 0]) ;; (dec (count start-string))
             i 0]
        ;; set up next input (single time step) by sampling from previous output
        (let [next-input (zeros [num-samples (input-columns iter)])]
          (dotimes [s num-samples]
            (let [output-prob-distribution (double-array (total-outcomes iter))]
              (dotimes [j (count output-prob-distribution)]
                (aset output-prob-distribution j (get-double output [s j])))
              (let [sampled-character-idx (sample-from-distribution output-prob-distribution rng)]
                ;; prepare next time step input
                (put-scalar next-input [s sampled-character-idx] 1.0)
                ;; add sampled character to stringbuilder (human readable output)
                (.append ^StringBuilder (nth sb s) (idx-to-char-map sampled-character-idx)))))
          (when (< i characters-to-sample)
            ;; do one time step of forward pass
            (recur (rnn-time-step net next-input)
                   (inc i)))))
    (map #(.toString ^StringBuilder %) sb))))

