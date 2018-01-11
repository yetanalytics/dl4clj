(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.html"}
    dl4clj.nn.multilayer.multi-layer-network
  (:require [dl4clj.utils :refer [contains-many? gensym* obj-or-code? eval-if-code]]
            [dl4clj.constants :as enum]
            [dl4clj.nn.api.model :refer [fit! init!]]
            [dl4clj.helpers :refer [new-lazy-iter reset-if-empty?! reset-iterator!]]
            [dl4clj.datasets.api.iterators :refer [has-next? next-example!]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [clojure.core.match :refer [match]])
  (:import [org.deeplearning4j.nn.multilayer MultiLayerNetwork]))

(defn new-multi-layer-network
  "constructor for a multi-layer-network given a config and optionaly
  some params (INDArray or vec).  The network is initialized "
  [& {:keys [conf params]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:conf (_ :guard seq?)
           :params (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         `(MultiLayerNetwork. ~conf (vec-or-matrix->indarray ~params))
         [{:conf _ :params _}]
         (let [[c p] (eval-if-code [conf seq?] [params seq?])]
           (MultiLayerNetwork. c (vec-or-matrix->indarray p)))
         [{:conf (_ :guard seq?)}]
         `(MultiLayerNetwork. ~conf)
         :else
         (MultiLayerNetwork. conf)))

(defn train-mln-with-ds-iter!
  "train the supplied multi layer network on the supplied dataset

  :iter (iterator), an iterator wrapping a dataset
   - see: dl4clj.datasets.iterators

  :n-epochs (int), the number of passes through the dataset"
  [& {:keys [mln iter n-epochs as-code?]
      :or {as-code? true}
      :as opts}]
  (let [n* (gensym* :sym "n-epochs")
        mln* (gensym* :sym "mln")]
    (match [opts]
           [{:mln (_ :guard seq?)
             :iter (_ :guard seq?)
             :n-epochs (:or (_ :guard number?)
                            (_ :guard seq?))}]
           ;; figure out whats going wrong here
           (obj-or-code?
            as-code?
            `(let [~mln* ~mln]
               (dotimes [~n* ~n-epochs]
                 (fit! :mln ~mln* :iter ~iter))
               ~mln*))
           :else
           (let [[model i n-e] (eval-if-code [mln seq?]
                                             [iter seq?]
                                             [n-epochs seq? number?])]
            (do (dotimes [n n-e]
                 (fit! :mln model :iter i))
               model)))))
