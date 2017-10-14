(ns dl4clj.core
  (:require [dl4clj.datasets.api.record-readers :as rr-api]
            [dl4clj.datasets.api.iterators :as iter-api]
            [dl4clj.datasets.api.pre-processors :as norm-api]
            [dl4clj.nn.multilayer.multi-layer-network :as mln]
            [dl4clj.nn.api.model :as m-api]
            #_[dl4clj.nn.api.multi-layer-network :as mln-api]
            [dl4clj.earlystopping.model-saver :as saver]
            [dl4clj.earlystopping.score-calc :as scorer]
            [dl4clj.earlystopping.early-stopping-config :as es-conf]
            [dl4clj.earlystopping.early-stopping-trainer :as es-trainer]
            [dl4clj.earlystopping.api.early-stopping-trainer :as trainer-api]
            [dl4clj.spark.dl4j-multi-layer :as spark-mln]
            [dl4clj.spark.api.dl4j-multi-layer :as spark-mln-api]
            [dl4clj.spark.data.java-rdd :as rdd]
            [dl4clj.utils :as u]
            [dl4clj.helpers :as h]
            [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; data
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn normalize-iter!
  "given an iterator and a normalizer, the normalizer is fit on the iterator
  and then the iterators data is normalized"
  [& {:keys [iter normalizer as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:iter (_ :guard seq?)
           :normalizer (_ :guard seq?)}]
         (let [i (u/gensym* :sym "the-iter")
               norm (u/gensym* :sym "the-normalizer")]
           (u/obj-or-code?
            as-code?
            `(let [~i ~iter
                   ~norm (norm-api/fit-iter! :normalizer ~normalizer
                                             :iter ~i)]
               (iter-api/set-pre-processor! :pre-processor ~norm
                                            :iter ~i))))
         [{:iter (_ :guard seq?)
           :normalizer _}]
         (let [iter* (eval iter)
               norm (norm-api/fit-iter! :normalizer normalizer
                                        :iter iter*)]
           (iter-api/set-pre-processor! :pre-processor norm
                                        :iter iter*))
         [{:iter _
           :normalizer (_ :guard seq?)}]
         (let [norm* (eval normalizer)
               norm (norm-api/fit-iter! :normalizer norm*
                                        :iter iter)]
           (iter-api/set-pre-processor! :pre-processor norm
                                        :iter iter))

         [{:iter _
           :normalizer _}]
         (let [norm (norm-api/fit-iter! :normalizer normalizer
                                        :iter iter)]
           (iter-api/set-pre-processor! :pre-processor norm
                                        :iter iter))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; model training
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; add in as code?
(defn model-from-conf
  "creates a multi layer network out of a configuration and initializes
   the model.  If provided code, will return code. If provided dl4j objects,
   will return the object"
  [& {:keys [nn-conf]}]
  ;; this is going to change
  (m-api/init! :model (mln/new-multi-layer-network :conf nn-conf)))

;; add regular training fn

(defn train-with-early-stopping
  "creates an optionally fits an early stopping trainer based on the supplied nn-conf."
  [& {:keys [nn-conf training-iter testing-iter eval-every-n-epochs model-saver
             iteration-termination-conditions epoch-termination-conditions
             save-last-model? average? train? return-best-model? as-code?]
      :or {model-saver (saver/new-in-memory-saver)
           save-last-model? true
           average? true
           train? true
           as-code? true
           return-best-model? true}
      :as opts}]
  ;; spec to make sure model-saver is a seq if passed
  (match [opts]
         [{:iteration-termination-conditions (_ :guard coll?)
           :epoch-termination-conditions (_ :guard coll?)
           :training-iter (_ :guard seq?)
           :testing-iter (_ :guard seq?)
           :eval-every-n-epochs (:or (_ :guard int?)
                                     (_ :guard seq?))
           :nn-conf (_ :guard seq?)}]
         (let [mln (model-from-conf :nn-conf nn-conf)
               ;; ^ might not need to initialize the model but I prob do
               score-calc (scorer/new-ds-loss-calculator :iter testing-iter
                                                         :average? average?)

               es-config (es-conf/new-early-stopping-config
                          ;; should check to see if these are already in a vector
                          :epoch-termination-conditions epoch-termination-conditions
                          :iteration-termination-conditions iteration-termination-conditions
                          :eval-every-n-epochs eval-every-n-epochs
                          :model-saver model-saver
                          :save-last-model? save-last-model?
                          :score-calculator score-calc)
               trainer (es-trainer/new-early-stopping-trainer
                        :early-stopping-conf es-config
                        :mln mln :iter training-iter)
               trained (if train?
                         (trainer-api/fit-trainer! trainer)
                         trainer)
               best-model (if (and return-best-model? train?)
                            (trainer-api/get-best-model-from-result
                             trained)
                            trained)]
           (u/obj-or-code? as-code? best-model))
         :else
         (throw (Exception. "you must supply args as code"))))

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
                n (u/gensym* :sym "number-epochs" :n 1)
                rdd (u/gensym* :sym "rdd" :n 1)]
            `(let [~evaled (spark-mln/new-spark-multi-layer-network
                            :spark-context ~spark-context
                            :mln ~mln-conf
                            :training-master ~training-master
                            :as-code? false)
                   ~sc (spark-mln-api/get-spark-context :spark-mln ~evaled)
                   ~rdd (.parallelize ~sc (h/data-from-iter
                                           :iter ~iter :as-code? false)
                                      (int ~n-slices))]
               (do (dotimes [~n ~n-epochs]
                     (doto ~evaled (.fit ~rdd)))
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
                n (u/gensym* :sym "number-epochs" :n 1)
                rdd (u/gensym* :sym "rdd" :n 1)
                i (u/gensym* :sym "inc" :n 1)
                result (u/gensym* :sym "place_holder" :n 1)]
           `(let [~evaled (spark-mln/new-spark-multi-layer-network
                          :spark-context ~spark-context
                          :mln ~mln-conf
                          :training-master ~training-master
                          :as-code? false)
                  ~sc (spark-mln-api/get-spark-context :spark-mln ~evaled)
                  ~rdd (.parallelize ~sc (h/data-from-iter :iter ~iter
                                                           :as-code? false))]
              (do (dotimes [~n ~n-epochs]
                    (doto ~evaled (.fit ~rdd)))
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
