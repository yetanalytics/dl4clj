(ns ^{:doc "

GravesLSTM Character modelling example

@author Joachim De Beule, based on Alex Black's java code, see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java and Andrej Karpathy's blog (http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and python code (https://gist.github.com/karpathy/d4dee566867f8291f086)

For general instructions using deeplearning4j's implementation of recurrent neural nets see http://deeplearning4j.org/usingrnns.html
"}
  dl4clj.examples.rnn.graves-lstm-char-modelling-example
  (:require [dl4clj.examples.example-utils :refer (shakespeare)]
            [dl4clj.examples.rnn.lr-character-iterator :refer (lr-character-iterator)]
            ;; [dl4clj.examples.example-utils :refer (+default-character-set+)]
            [dl4clj.nn.conf.layers.graves-lstm]
            [dl4clj.nn.conf.layers.rnn-output-layer]
            [nd4clj.linalg.factory.nd4j :refer (zeros set-enforce-numerical-stability!)]
            [nd4clj.linalg.api.ndarray.indarray :refer (put-scalar get-double tensor-along-dimension slice data)]
            [dl4clj.nn.conf.distribution.normal-distribution]
            [nd4clj.linalg.lossfunctions.loss-functions]
            [dl4clj.nn.conf.neural-net-configuration :refer (neural-net-configuration)]
            [dl4clj.nn.multilayer.multi-layer-network :refer (multi-layer-network init get-layers get-layer rnn-clear-previous-state rnn-time-step)]
            [dl4clj.nn.api.model :as model]
            [dl4clj.nn.api.classifier :as classifier]
            [dl4clj.nn.api.layer]))

(defn graves-lstm-char-modeling-net [iter hidden-layer-size]
  (let [opts {:optimization-algo :stochastic-gradient-descent
              :iterations 1
              :learning-rate 0.1
              :rms-decay 0.95
              :seed 12345
              :regularization true
              :l2 0.001
              :list 3
              :layers {0 {:graves-lstm
                          {:n-in (count (:valid-chars iter))
                           :n-out hidden-layer-size
                           :updater :rmsprop
                           :activation :tanh
                           :weight-init :distribution
                           :dist {:normal {:mean 0.0, :std 0.01}}}}
                       1 {:graves-lstm
                          {:n-in hidden-layer-size
                           :n-out hidden-layer-size
                           :updater :rmsprop
                           :activation :tanh
                           :weight-init :distribution
                           :dist {:normal {:mean 0.0, :std 0.01}}}}
                       2 {:rnnoutput
                          {:loss-function :mcxent
                           :n-in hidden-layer-size
                           :n-out (count (:valid-chars iter))
                           :updater :rmsprop
                           :activation :softmax
                           :weight-init :distribution
                           :dist {:normal {:mean 0.0, :std 0.01}}}}}
              :pretrain false
              :backprop true}
        net (multi-layer-network (neural-net-configuration opts))]
    (init net)
    (dotimes [i (count (get-layers net))]
      (println "Number of parameters in layer "  i  ": "  (model/num-params (get-layer net i))))
    (println "Total number of network parameters: " (reduce + (map model/num-params (get-layers net))))
    net))

(defn- sample-from-distribution 
  "Sample from a probability distribution over discrete classes given as a vector of probabilities
  summing to 1.0."
  [distribution] 
  (let [toss (rand)]
    (loop [i 0
           sum (nth distribution 0)]
      (cond (<= toss sum) i
            (< i (count distribution)) (recur (inc i) (+ sum (nth distribution i)))
            :else (throw (IllegalArgumentException. (str "Distribution is invalid? toss= " toss ", sum=" sum)))))))

(defn sample [net iter start-string num-samples sample-length]
  (let [char-to-idx (:char-to-idx iter)
        idx-to-char (zipmap (vals char-to-idx) (keys char-to-idx))
        initialization-input (zeros [num-samples (count char-to-idx) (count start-string)])]    
    ;; Fill input with data from start-string
    (dotimes [i (count start-string)]
      (let [idx (char-to-idx (nth start-string i))]
        (dotimes [j num-samples]
          (put-scalar initialization-input [j idx i] 1.0))))
    
    (rnn-clear-previous-state net)

    (loop [samples (for [i (range num-samples)] start-string)
           output (tensor-along-dimension (rnn-time-step net initialization-input)
                                          (dec (count start-string))
                                          [1 0])]
      ;; Set up next input (single time step) by sampling from previous output
      (let [next-input (zeros num-samples (count char-to-idx))
            next-samples (for [s (range num-samples)]
                           (let [sampled-character-idx (sample-from-distribution (data (slice output s 1)))]
                             (put-scalar next-input [s sampled-character-idx] 1.0)
                             (str (nth samples s)  (idx-to-char sampled-character-idx))))]
        (if (< (count (first samples)) sample-length)
          (recur next-samples (rnn-time-step net next-input))
          next-samples)))))

(comment
  
  ;; custom dataset iterator:
  (def iter (lr-character-iterator (shakespeare)))

  (def net (graves-lstm-char-modeling-net iter 100))

  (dotimes [i 10]
    (.reset iter)
    (classifier/fit net iter)
    (println "finished iteration" i)
    (doseq [s (sample net iter "a" 10 80)]
      (println "sample:" s)))

  )
