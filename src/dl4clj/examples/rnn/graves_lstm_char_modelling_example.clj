(ns ^{:doc "

GravesLSTM Character modelling example

@author Joachim De Beule, based on Alex Black's java code, see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java and Andrej Karpathy's blog (http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and python code (https://gist.github.com/karpathy/d4dee566867f8291f086)

For general instructions using deeplearning4j's implementation of recurrent neural nets see http://deeplearning4j.org/usingrnns.html
"}
  dl4clj.examples.rnn.graves-lstm-char-modelling-example
  (:require [dl4clj.examples.rnn.tools :refer [sample-characters-from-network]]
            [dl4clj.examples.example-utils :refer (shakespeare)]
            [dl4clj.examples.rnn.character-iterator :refer (get-shakespeare-iterator)]
            [nd4clj.linalg.dataset.api.iterator.data-set-iterator :refer (input-columns total-outcomes reset)]
            [dl4clj.nn.conf.layers.graves-lstm]
            [dl4clj.nn.conf.layers.rnn-output-layer]
            [dl4clj.nn.conf.distribution.uniform-distribution]
            [nd4clj.linalg.lossfunctions.loss-functions]
            [dl4clj.nn.conf.neural-net-configuration :refer (neural-net-configuration)]
            [dl4clj.nn.multilayer.multi-layer-network :refer (multi-layer-network init get-layers get-layer)]
            [dl4clj.nn.api.model :as model]
            [dl4clj.nn.api.classifier :as classifier]
            [dl4clj.nn.api.layer])
  (:import [java.util Random]))


(def mini-batch-size 32)
(def example-length 100)
(def examples-per-epoch (* mini-batch-size 50))
(def lstm-layer-size 200)
(def num-epochs 30)
(def generation-initialization nil)
(def rng (Random. 12345))
(def n-characters-to-sample 300)
(def n-samples-to-generate 4)

;; Get a DataSetIterator that handles vectorization of text into something we can use to train our
;; GravesLSTM network.
(def iter (get-shakespeare-iterator mini-batch-size example-length examples-per-epoch))

;; Set up network configuration:
;; broken
(def conf (neural-net-configuration
           {:optimization-algo :stochastic-gradient-descent
            :iterations 1
            :learning-rate 0.1
            :rms-decay 0.95
            :seed 12345
            :regularization true
            :l2 0.001
            :list 3
            :layers {0 {:graves-lstm
                        {:n-in (input-columns iter)
                         :n-out lstm-layer-size
                         :updater :rmsprop
                         :activation :tanh
                         :weight-init :distribution
                         :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                     1 {:graves-lstm
                        {:n-in lstm-layer-size
                         :n-out lstm-layer-size
                         :updater :rmsprop
                         :activation :tanh
                         :weight-init :distribution
                         :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                     2 {:rnnoutput
                        {:loss-function :mcxent
                         :activation :softmax
                         :updater :rmsprop
                         :n-in lstm-layer-size
                         :n-out (total-outcomes iter)
                         :weight-init :distribution
                         :dist {:uniform {:lower -0.08, :upper 0.08}}}}}
            :pretrain false
            :backprop true}))
(def net (multi-layer-network conf))
(init net)
;; not yet implemented:
;; net.setListeners(new ScoreIterationListener(1));

;; Print the  number of parameters in the network (and for each layer)
(dotimes [i (count (get-layers net))]
  (println "Number of parameters in layer "  i  ": "  (model/num-params (get-layer net i))))
(println "Total number of network parameters: " (reduce + (map model/num-params (get-layers net))))

;; Do training, and then generate and print samples from network
(dotimes [i num-epochs]
  (classifier/fit net iter)

  (println "--------------------")
  (println "Completed epoch " i )
  (println "Sampling characters from network given initialization \"\"")
  (loop [j 0
         samples (sample-characters-from-network generation-initialization net iter rng n-characters-to-sample n-samples-to-generate)]
    (println "----- Sample " j " -----")
    (println (first samples))
    (println)
    (when-not (empty? samples)
      (recur (inc j) (rest samples))))

  ;; Reset iterator for another epoch
  (reset iter))
