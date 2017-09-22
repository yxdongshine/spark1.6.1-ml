package src.main.yxd_ml.sparkStreamingMLlib

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans

/**
  * Created by 11725 on 2017/6/4.
  * 新闻的数据
  * title,time,author,context
  * *******************************文章都是要经过分词****************************
  */
object NewsTrain {

  def main(args: Array[String]) {


    val conf= new SparkConf()
    val sc=new SparkContext(conf)
    /**
     * k-means 参考spark 1.6.1官网code：
     * http://spark.apache.org/docs/1.6.1/mllib-clustering.html
     */
    //我们对内容分类，是要取第三个参数context
    val newsrdd=sc.textFile("spark01:8080/log/news")
      .map(s => Vectors.dense(s.split(" ").map( arr => arr(3).toDouble)))

    //聚类导入模块
    val parsedata=newsrdd
    val numclusters=14  //有6个聚类中
    val numIterors=20 //设置迭代次数
    //训练我们的模型
    val clusters=KMeans.train(parsedata,numclusters,numIterors)

    //打印出聚类中
    val clustercenters=clusters.clusterCenters

    //聚类结果标签,将原有的数据通过聚类找到中心点判断是哪些类别，并将文本标注出类别
    val labels=clusters.predict(parsedata)

    //保存模
    clusters.save(sc,"xxxxx")
  }

}
