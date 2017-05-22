(ns ^{:doc "A general purpose interface for evaluating neural networks

see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/IEvaluation.html"}
    dl4clj.eval.interface.i-evaluation
  (:import [org.deeplearning4j.eval IEvaluation]))

(defn merge!
  "merges objects that implemented the IEvaluation interface

  evaler and other-evaler can be evaluations, ROCs or MultiClassRocs"
  [& {:keys [evaler other-evaler]}]
  (doto evaler (.merge other-evaler)))

(defn eval-time-series!
  "evalatues a time series given labels and predictions.

  labels-mask is optional and only applies when there is a mask"
  [& {:keys [labels predicted labels-mask evaler]
      :as opts}]
  (cond (contains? opts :labels-mask)
        (doto evaler (.evalTimeSeries labels predicted labels-mask))
        (false? (contains? opts :labels-mask))
        (doto evaler (.evalTimeSeries labels predicted))
        :else
        (assert false "you must supply labels-mask and/or labels and predicted values")))

(defn eval-classification!
  "depending on args supplied in opts map, does one of:

  - Collects statistics on the real outcomes vs the guesses.
  - Evaluate the output using the given true labels, the input to the multi layer network and the multi layer network to use for evaluation
  - Evaluate the network, with optional metadata
  - Evaluate a single prediction (one prediction at a time)

  1) is accomplished by supplying :real-outcomes and :guesses
  2) is accomplished by supplying :true-labels, :in and :comp-graph or :mln
  3) is accomplished by supplying :real-outcomes, :guesses and :record-meta-data
  4) is accomplished by supplying :predicted-idx and :actual-idx"
  [& {:keys [real-outcomes guesses
             true-labels in comp-graph
             record-meta-data mln
             predicted-idx actual-idx evaler]
      :as opts}]
  (assert (contains? opts :evaler) "you must provide an evaler to evaluate a classification task")
  (cond (contains-many? opts :true-labels
                        :in :comp-graph)
        (doto evaler (.eval true-labels in comp-graph))
        (contains-many? opts :true-labels
                        :in :mln)
        (doto evaler (.eval true-labels in mln))
        (contains-many? opts :real-outcomes
                        :guesses :record-meta-data)
        (doto evaler (.eval real-outcomes guesses record-meta-data))
        (contains-many? opts :real-outcomes
                        :guesses)
        (doto evaler (.eval real-outcomes guesses))
        (contains-many? opts :predicted-idx :actual-idx)
        (doto evaler (.eval predicted-idx actual-idx))
        :else
        (assert false "you must supply the evaler one of the set of opts described in the doc string")))
