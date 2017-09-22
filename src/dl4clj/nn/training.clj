(ns dl4clj.nn.training
  (:require [dl4clj.utils :as u]
            [dl4clj.spark.dl4j-multi-layer :as spark-mln]
            [dl4clj.spark.api.dl4j-multi-layer :as spark-mln-api]
            [dl4clj.helpers :as h]
            [clojure.core.match :refer [match]]
            [dl4clj.spark.data.java-rdd :as rdd]))

(defn train-with-spark
  [& {:keys [spark-context mln-conf training-master
             iter n-epochs rdd n-slices as-code?]
      :or {as-code? true
           n-epochs 10}
      :as opts}]
  (match [opts]
         ;; specified number of slices for .parallelize
         [{:spark-context (_ :guard seq?)
           :mln-conf (_ :guard seq?)
           :training-master (_ :guard seq?)
           :iter (_ :guard seq?)
           :n-slices (:or (_ :guard number?)
                          (_ :guard seq?))}]
         (u/obj-or-code?
          as-code?
          (let [evaled (u/gensym* :sym "evaled-nn" :n 1)
                sc (u/gensym* :sym "spark-context" :n 1)
                n (u/gensym* :sym "number-epochs" :n 1)]
            `(let [~evaled (spark-mln/new-spark-multi-layer-network
                            :spark-context ~spark-context
                            :mln ~mln-conf
                            :training-master ~training-master
                            :as-code? false)
                   ~sc (spark-mln-api/get-spark-context :spark-mln ~evaled)]
               (do
                 (dotimes [~n ~n-epochs]
                   (doto ~evaled
                     (.fit
                      (.parallelize
                       ~sc
                       (h/data-from-iter :iter ~iter :as-code? false)
                       (int ~n-slices)))))
                 ~evaled))))
         ;; didnt specified number of slices for .parallelize
         [{:spark-context (_ :guard seq?)
           :mln-conf (_ :guard seq?)
           :training-master (_ :guard seq?)
           :iter (_ :guard seq?)}]
         (u/obj-or-code?
          as-code?
          (let [evaled (u/gensym* :sym "evaled-nn" :n 1)
                sc (u/gensym* :sym "spark-context" :n 1)
                n (u/gensym* :sym "number-epochs" :n 1)]
           `(let [~evaled (spark-mln/new-spark-multi-layer-network
                          :spark-context ~spark-context
                          :mln ~mln-conf
                          :training-master ~training-master
                          :as-code? false)
                 ~sc (spark-mln-api/get-spark-context :spark-mln ~evaled)]
              (do
                (dotimes [~n ~n-epochs]
                  (.fit ~evaled
                        (.parallelize ~sc (h/data-from-iter :iter ~iter
                                                            :as-code? false))))
                ~evaled))))
         [{:spark-context _
           :mln-conf _
           :training-master _
           :iter _
           :n-slices _}]
         ;; specified number of slices for .parallelize
         (let [spark-mln (spark-mln/new-spark-multi-layer-network
                          :spark-context spark-context
                          :mln-conf mln-conf
                          :training-master training-master)
               rdd (rdd/java-rdd-from-iter :spark-context spark-context
                                           :iter iter
                                           :num-slices n-slices)]
           (spark-mln-api/fit-spark-mln! :spark-mln spark-mln
                                         :rdd rdd
                                         :n-epochs n-epochs))
         [{:spark-context _
           :mln-conf _
           :training-master _
           :iter _}]
         ;; didnt specified number of slices for .parallelize
         (let [spark-mln (spark-mln/new-spark-multi-layer-network
                          :spark-context spark-context
                          :mln-conf mln-conf
                          :training-master training-master)
               rdd (rdd/java-rdd-from-iter :spark-context spark-context
                                           :iter iter)]
           (spark-mln-api/fit-spark-mln! :spark-mln spark-mln
                                         :rdd rdd
                                         :n-epochs n-epochs))
         [{:spark-context _
           :mln-conf _
           :training-master _
           :rdd _}]
         ;; provided an rdd instead of an iterator
         (let [spark-mln (spark-mln/new-spark-multi-layer-network
                          :spark-context spark-context
                          :mln-conf mln-conf
                          :training-master training-master)]
           (spark-mln-api/fit-spark-mln! :spark-mln spark-mln
                                         :rdd rdd
                                         :n-epochs n-epochs))))
