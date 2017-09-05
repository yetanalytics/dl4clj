(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.html"}
    dl4clj.nn.multilayer.multi-layer-network
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.constants :as enum]
            [dl4clj.nn.api.model :refer [fit! init!]]
            [dl4clj.helpers :refer [new-lazy-iter reset-if-empty?! reset-iterator!]]
            [dl4clj.datasets.api.iterators :refer [has-next? next-example!]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [clojure.core.match :refer [match]])
  (:import [org.deeplearning4j.nn.multilayer MultiLayerNetwork]))

(defn new-multi-layer-network
  "constructor for a multi-layer-network given a config and optionaly
  some params (INDArray or vec)"
  [& {:keys [conf params]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:conf (_ :guard seq?) :params _}]
         `(MultiLayerNetwork. ~conf (vec-or-matrix->indarray ~params))
         [{:conf _ :params _}]
         (MultiLayerNetwork. conf (vec-or-matrix->indarray params))
         [{:conf (_ :guard seq?)}]
         `(MultiLayerNetwork. ~conf)
         :else
         (MultiLayerNetwork. conf)))

;; move to core
(defn train-mln-with-ds-iter!
  "train the supplied multi layer network on the supplied dataset

  :iter (iterator), an iterator wrapping a dataset
   - see: dl4clj.datasets.iterators

  :n-epochs (int), the number of passes through the dataset"
  [& {:keys [mln iter n-epochs]}]
  (dotimes [n n-epochs]
    (while (has-next? iter)
      ;; fit handles the iter resetting
      (fit! :mln mln :iter iter)))
  mln)

(defn train-mln-with-lazy-seq!
  "train the supplied multi layer network on the dataset contained within
   the supplied lazy seq

   :lazy-seq-data (lazy-seq), a lazy-seq of dataset objects
    - created by data-from-iter in: dl4clj.helpers"

  ;; test this
  [& {:keys [lazy-seq-data mln n-epochs]}]
  (dotimes [n n-epochs]
    (loop [_ mln
           accum! lazy-seq-data]
      ;; this could never complete
      ;; rest always returns a seq
      (if (not (empty? accum!))
        (let [data (first accum!)]
          (recur (fit! :mln mln :data data)
                 (rest accum!)))
        mln)))

  #_(dotimes [n n-epochs]
    ;; look into avoiding creation of lazy iter and just recursively going through
    ;; lazy seq

    ;; dont know of a more effecient way of doing this
    ;; you cant reset a lazy-seq-iter so i just make a new one
    ;; prob why training takes non-neglegible amount of time



    #_(let [iter (new-lazy-iter lazy-seq-data)]
      (while (has-next? iter)
        (let [nxt (next-example! iter)]
          (fit! :mln mln :data nxt))))))
