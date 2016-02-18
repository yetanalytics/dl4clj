(ns ^{:doc "

GravesLSTM Character modelling example

@author Joachim De Beule, based on Alex Black's java code, see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java

For general instructions using deeplearning4j's implementation of recurrent neural nets see http://deeplearning4j.org/usingrnns.html
"}
  dl4clj.examples.rnn.graves-lstm-char-modelling-example
  (:require [dl4clj.examples.example-utils :refer (shakespeare)]
            [dl4clj.examples.rnn.lr-character-iterator :refer (lr-character-iterator)]
            ;; [dl4clj.examples.example-utils :refer (+default-character-set+)]
            [dl4clj.nn.conf.layers.graves-lstm]
            [dl4clj.nn.conf.layers.rnn-output-layer]
            [nd4clj.linalg.factory.nd4j :refer (zeros set-enforce-numerical-stability!)]
            [nd4clj.linalg.api.ndarray.indarray :refer (put-scalar get-double tensor-along-dimension slice)]
            [dl4clj.nn.conf.distribution.normal-distribution]
            [dl4clj.nn.conf.distribution.uniform-distribution]
            [nd4clj.linalg.lossfunctions.loss-functions]
            [dl4clj.nn.conf.neural-net-configuration :refer (neural-net-configuration)]
            [dl4clj.nn.multilayer.multi-layer-network :refer (multi-layer-network init get-layers get-layer rnn-clear-previous-state rnn-time-step)]
            [dl4clj.nn.api.model :as model]
            [dl4clj.nn.api.classifier :as classifier]
            [dl4clj.nn.api.layer])
  (:import [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.factory Nd4j]
           ;; [java.io File IOException]
           ;; [java.net URL]
           ;; [java.text CharacterIterator]
           ;; [java.nio.charset Charset]
           ;; [java.util Random]
           
           [org.deeplearning4j.datasets.iterator DataSetIterator]

           [org.deeplearning4j.nn.api Layer OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration Updater NeuralNetConfiguration NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.conf.distribution UniformDistribution]
           [org.deeplearning4j.nn.conf.layers GravesLSTM GravesLSTM$Builder RnnOutputLayer RnnOutputLayer$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]
           ))

;;; There are two ways to setup an rnn:
;;; 1) via `(multi-layer-network (neural-net-configuration opts))`:

(defn graves-lstm-char-modeling-net-1 [iter hidden-layer-size]
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
                           :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                       1 {:graves-lstm
                          {:n-in hidden-layer-size
                           :n-out hidden-layer-size
                           :updater :rmsprop
                           :activation :tanh
                           :weight-init :distribution
                           :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                       2 {:rnnoutput
                          {:loss-function :mcxent
                           :n-in hidden-layer-size
                           :n-out (count (:valid-chars iter))
                           :updater :rmsprop
                           :activation :softmax
                           :weight-init :distribution
                           :dist {:uniform {:lower -0.08, :upper 0.08}}}}}
              :pretrain false
              :backprop true}
        net (multi-layer-network (neural-net-configuration opts))]
    (init net)
    (dotimes [i (count (get-layers net))]
      (println "Number of parameters in layer "  i  ": "  (model/num-params (get-layer net i))))
    (println "Total number of network parameters: " (reduce + (map model/num-params (get-layers net))))
    net))

;;; 2) via `(multi-layer-network ml-opts)`

(defn graves-lstm-char-modeling-net-2 [iter hidden-layer-size]
  (let [ml-opts {:input-pre-processors {},
                 :confs
                 [{:num-iterations 5,
                   :step-function nil,
                   :use-drop-connect false,
                   :mini-batch true,
                   :variables [],
                   :seed 12345,
                   :max-num-line-search-iterations 5,
                   :use-regularization true,
                   :layer
                   {:graves-lstm
                    {:l1 0.0,
                     :drop-out 0.0,
                     :dist {:uniform {:lower -0.08, :upper 0.08}},
                     :rho 0.0,
                     :forget-gate-bias-init 1.0,
                     :activation-function "tanh",
                     :learning-rate-after {},
                     :gradient-normalization "None",
                     :weight-init "DISTRIBUTION",
                     :nout hidden-layer-size,
                     :adam-var-decay 0.999,
                     :bias-init 0.0,
                     :lr-score-based-decay 0.0,
                     :momentum-after {},
                     :l2 0.001,
                     :updater "RMSPROP",
                     :momentum 0.5,
                     :layer-name "genisys",
                     :nin 86,
                     :learning-rate 0.1,
                     :adam-mean-decay 0.9,
                     :rms-decay 0.95,
                     :gradient-normalization-threshold 1.0}},
                   :use-schedules false,
                   :minimize true,
                   :optimization-algo "STOCHASTIC_GRADIENT_DESCENT",
                   :time-series-length 1}
                  {:num-iterations 5,
                   :step-function nil,
                   :use-drop-connect false,
                   :mini-batch true,
                   :variables [],
                   :seed 12345,
                   :max-num-line-search-iterations 5,
                   :use-regularization true,
                   :layer
                   {:graves-lstm
                    {:l1 0.0,
                     :drop-out 0.0,
                     :dist {:uniform {:lower -0.08, :upper 0.08}},
                     :rho 0.0,
                     :forget-gate-bias-init 1.0,
                     :activation-function "tanh",
                     :learning-rate-after {},
                     :gradient-normalization "None",
                     :weight-init "DISTRIBUTION",
                     :nout hidden-layer-size,
                     :adam-var-decay 0.999,
                     :bias-init 0.0,
                     :lr-score-based-decay 0.0,
                     :momentum-after {},
                     :l2 0.001,
                     :updater "RMSPROP",
                     :momentum 0.5,
                     :layer-name "genisys",
                     :nin hidden-layer-size,
                     :learning-rate 0.1,
                     :adam-mean-decay 0.9,
                     :rms-decay 0.95,
                     :gradient-normalization-threshold 1.0}},
                   :use-schedules false,
                   :minimize true,
                   :optimization-algo "STOCHASTIC_GRADIENT_DESCENT",
                   :time-series-length 1}
                  {:num-iterations 5,
                   :step-function nil,
                   :use-drop-connect false,
                   :mini-batch true,
                   :variables [],
                   :seed 12345,
                   :max-num-line-search-iterations 5,
                   :use-regularization true,
                   :layer
                   {:rnnoutput
                    {:l1 0.0,
                     :drop-out 0.0,
                     :custom-loss-function nil,
                     :dist {:uniform {:lower -0.08, :upper 0.08}},
                     :rho 0.0,
                     :activation-function "softmax",
                     :learning-rate-after {},
                     :gradient-normalization "None",
                     :weight-init "DISTRIBUTION",
                     :nout 86,
                     :adam-var-decay 0.999,
                     :bias-init 0.0,
                     :lr-score-based-decay 0.0,
                     :momentum-after {},
                     :loss-function "MCXENT",
                     :l2 0.001,
                     :updater "RMSPROP",
                     :momentum 0.5,
                     :layer-name "genisys",
                     :nin hidden-layer-size,
                     :learning-rate 0.1,
                     :adam-mean-decay 0.9,
                     :rms-decay 0.95,
                     :gradient-normalization-threshold 1.0}},
                   :use-schedules false,
                   :minimize true,
                   :optimization-algo "STOCHASTIC_GRADIENT_DESCENT",
                   :time-series-length 1}],
                 :backprop-type "Standard",
                 :tbptt-back-length 20,
                 :redistribute-params false,
                 :pretrain false,
                 :tbptt-fwd-length 20,
                 :damping-factor 100.0,
                 :backprop true}
        net (multi-layer-network ml-opts)]
    (init net)
    (dotimes [i (count (get-layers net))]
      (println "Number of parameters in layer "  i  ": "  (model/num-params (get-layer net i))))
    (println "Total number of network parameters: " (reduce + (map model/num-params (get-layers net))))
    net))

(defn- output-distribution 
  "Utility fn to convert a 1 dimensional NDArray to an array of doubles."
  [^INDArray output]
  (let [d (double-array (.length output))]
    (dotimes [i (.length output)]
      (aset d i ^Double (get-double output [i])))
    d))

(defn- sample-from-distribution 
  "Given a probability distribution over discrete classes (an array of doubles), sample from the
  distribution and return the generated class index.
  @param distribution Probability distribution over classes. Must sum to 1.0.
"
  [^"[D" distribution] 
  (let [toss (rand)]
    (loop [i 0
           sum (aget distribution 0)]
      (cond (<= toss sum) i
            (< i (count distribution)) (recur (inc i) (+ sum (aget distribution i)))
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
                           (let [sampled-character-idx (sample-from-distribution (output-distribution (slice output s 1)))]
                             (put-scalar next-input [s sampled-character-idx] 1.0)
                             (str (nth samples s)  (idx-to-char sampled-character-idx))))]
        (if (< (count (first samples)) sample-length)
          (recur next-samples (rnn-time-step net next-input))
          next-samples)))))

(comment

  ;; custom dataset iterator:
  (def iter (lr-character-iterator (subs (shakespeare) 0 50100)))

  ;; There are two ways to make the same network:
  ;; via multi-layer-network applied to a neural-net-configuration
  (def net (graves-lstm-char-modeling-net-1 iter 50))
  ;; or via a multi-layer-network applied to a multi-layer-configuration
  ;; (def net (graves-lstm-char-modeling-net-2 iter 50))
  
  ;; fit net to data:
  (.reset iter)
  (classifier/fit net iter)
  
  ;; sample:
  (sample net iter "h" 5 20)
  
  ;; repeat...


  )
