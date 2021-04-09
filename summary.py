import re
import math
import numpy as np
from typing import *
import warnings


class SummaryExtractor(object):
    def __init__(
        self, 
        sent_tokenizer: Union[Callable[[str], List[str]], str] = r"。|？|！|；", 
        word_tokenizer: Union[Callable[[str], List[str]], str] = "jieba", 
        stopwords: List[str] = [],
        max_iter: int = 100,
        min_delta: float = 1e-5,
        return_importance: bool = False,
        eval_sim: Optional[Callable[[str, str], float]]=None
    ):
        self._sent_tokenizer = lambda x: re.split(sent_tokenizer, x) \
            if isinstance(sent_tokenizer, str) else sent_tokenizer

        if word_tokenizer == "jieba":
            import jieba
            self._word_tokenzier = jieba.cut
        else:
            self._word_tokenzier = word_tokenizer

        self._eval_sim = eval_sim if eval_sim else self._default_eval_sim

        self._stopwords = set(stopwords)
        self._max_iter = max_iter
        self._min_delta = min_delta
        self._return_importance = return_importance
    
    def _default_eval_sim(
        self, 
        sent1: str,
        sent2: str
    ) -> float:
        sent1 = [word for word in self._word_tokenzier(sent1) if word not in self._stopwords]
        sent2 = [word for word in self._word_tokenzier(sent2) if word not in self._stopwords]
        numerator = len(set(sent1) & set(sent2))
        denominator = math.log(len(sent1)+1e-6) + math.log(len(sent2)+1e-6)
        if numerator < 1e-12 or denominator < 1e-12:
            similarity = 0
        else:
            similarity = numerator / denominator
        return similarity
    

    def _eval_similarity_matrix(
        self, 
        sents: List[str]
    ):
        sent_num = len(sents)
        sent_similarity_matrix = np.zeros((sent_num, sent_num))
        for i in range(sent_num):
            for j in range(i, sent_num):
                # compute similarity between sent_i and sent_j
                sent_i = sents[i]
                sent_j = sents[j]
                similarity = self._eval_sim(sent_i, sent_j)
                sent_similarity_matrix[i, j] = similarity
                sent_similarity_matrix[j, i] = similarity
        return sent_similarity_matrix

    
    def _topk(
        self, 
        output: Tuple[str, float],
        k: int
    ) -> Union[List[str], List[Tuple[str, float]]]:
        sorted_output = sorted(output, key=lambda x:x[1], reverse=True)
        topk_importance = set([pair[1] for pair in sorted_output[:k]])
        if self._return_importance:
            return [pair for pair in output if pair[1] in topk_importance]
        else:
            return [pair[0] for pair in output if pair[1] in topk_importance]


class TextRank(SummaryExtractor):

    def summarize_document(
        self, 
        text: str, 
        k: int,
        d: float = 0.85
    ) -> Union[List[str], List[Tuple[str, float]]]:
        k = k if k > 1 else 1

        sents = self._sent_tokenizer(text)
        sent_num = len(sents)

        # get similarity matrix of sentences
        sim_matrix = self._eval_similarity_matrix(sents)
        sents_importance = self._pagerank(sim_matrix, d=d)
        
        output = [(sents[i], sents_importance[i]) for i in range(sent_num)]

        # select topk output
        return self._topk(output, k)

    def _pagerank(self, sim_matrix: np.ndarray, d: float = 0.85) -> np.ndarray:
        n = len(sim_matrix)
        importance = np.array([1/n for _ in range(n)])
        sim_matrix *= 1 - np.eye(n)

        for _ in range(self._max_iter):
            abs_delta = 0
            importance_last = importance.copy()
            for i in range(n):

                # compute ws(v_i) = 1-d + d * sum_over_vj( w_ij/(sum_over_vk(w_jk)) * ws(v_j) )
                importance[i] = np.sum(sim_matrix[i] / (np.sum(sim_matrix, axis=0)+1e-7) * importance_last) * d + (1-d)

            abs_delta = np.sum(abs(importance_last - importance))

            # convergence
            if abs_delta < self._min_delta:
                break

        return importance


class MMR(SummaryExtractor):
    
    def summarize_document(
        self, 
        text: str, 
        k: int,
        mmr_lambda: float = 0.9
    ) -> Union[List[str], List[Tuple[str, float]]]:

        k = k if k > 1 else 1
        sents = self._sent_tokenizer(text)
        if self._return_importance:
            warnings.warn("MMR module does not support 'return_importance' yet.")
        return self._mmr(sents, k, mmr_lambda)
    

    def _mmr(
        self,
        sents: List[str],
        k: int,
        mmr_lambda: float = 0.9
    ) -> Union[List[str], List[Tuple[str, float]]]:
        text = "。".join(sents)

        # evaluate similarity score between total text and sentence

        scores = {sent: self._eval_sim(text, sent) for sent in sents}
        
        output = set()
        while k > 0:
            importance = {}
            for sent in scores.keys():
                if sent not in output:
                    importance[sent] = mmr_lambda * scores[sent] - (1-mmr_lambda) * self._eval_sim(sent, "。".join(list(output)))
            if len(importance) > 0:

                seleted_sent = max(importance.items(), key=lambda x:x[1])[0]
                output.add(seleted_sent)
                k -= 1
            else: 
                break
        
        # retain origin order
        return [sent for sent in sents if sent in output]



if __name__ == "__main__":
    text = "中新网1月11日电据商务部网站消息，商务部办公厅近日印发通知，提出重点地区的电商企业要逐步停止使用不可降解的塑料包装袋、一次性塑料编织袋，减少使用不可降解塑料胶带。  \
        资料图：工作人员分拣快递。\
        中新社记者张宇摄 通知从持续推动电商企业节能增效、协同推进快递包装绿色供应链管理、发挥平台优势培育绿色发展生态、保障措施四方面提出了多项要求，具体如下： 持续推动电商企业节能增效 通知提出，强化绿色发展理念。\
        加强法规政策宣贯，引导电商企业牢固树立生态环境保护主体责任意识，把生态环境保护和可持续发展作为企业发展的基本准则，严格遵守《电子商务法》《固体废物污染环境防治法》及塑料污染治理、快递包装绿色转型等法律法规和政策标准要求，合法合规经营。\
        鼓励电商企业积极履行生态环境保护社会责任，建立统计评估制度，主动公开一次性塑料包装减量替代方案，自觉接受公众和社会监督。 提升低碳环保水平。\
        引导电商企业积极应用节能环保先进技术和设备，做好办公经营场所、网络机房、仓储和物流配送的绿色建设和改造，建立健全绿色运营体系。\
        组织开展国家绿色数据中心推荐，鼓励大型电商企业加强绿色数据中心建设，选择能效水平高、技术先进、管理完善的数据中心争创国家绿色数据中心。\
        引导电商产业园区向绿色低碳循环方向改造升级，探索合同能源管理，推进园区优化环境管理、公共设施共建共享、资源循环利用，带动园区企业绿色发展。\
        增强数字化运营能力。在常态化疫情防控下，鼓励电商企业继续采用远程办公、视频会议、网上签约等低碳运营方式，提高线上线下协同工作效率。\
        大力推广电子发票、电子合同，推动电商企业实行电子发票报销入账归档全流程电子化管理。\
        鼓励电商企业应用大数据、云计算、人工智能等现代信息技术，加强供需匹配，提高库存周转率，推动多渠道物流共享，应用科学配载，降低物流成本和能耗。 \
        协同推进快递包装绿色供应链管理 通知强调，推动快递包装减量。鼓励电商企业通过产地直采、原装直发、聚单直发等模式，减少快递包装用量。\
        引导电商企业与商品生产企业合作，设计应用满足快递物流配送需求的商品包装，减少商品在快递环节的二次包装。\
        引导电商企业与快递企业合作，提高包装产品与寄递物品的匹配度，共同落实快递包装相关标准和规范，减少过度包装、随意包装，加大低克重高强度快递包装纸箱、免胶纸箱用量。\
        推进可循环包装应用。\
        鼓励电商企业参与快递包装循环利用新技术、新产品研发并积极推广应用，与快递企业、商业机构、便利店和物业服务企业等创新合作，建立可循环快递包装回收渠道，探索建立上下游衔接、跨平台互认的运营体系。\
        推动生鲜电商企业在同城配送中推广应用可循环配送箱、可复用冷藏式快递箱等。 规范快递包装使用。加强规范引导，落实好电商领域快递包装禁限制要求。\
        重点地区的电商企业要逐步停止使用不可降解的塑料包装袋、一次性塑料编织袋，减少使用不可降解塑料胶带。\
        推动电商外卖平台与回收企业等开展多方合作，在重点区域投放外卖餐盒等回收设施。\
        推动电商企业建立包装产品合格供应商制度，扩大合格供应商包装产品采购比例，禁止采购使用重金属含量、溶剂残留等超标的劣质包装和有毒有害材料制成的填充物。 \
        发挥平台优势培育绿色发展生态 通知指出，落实电商平台绿色管理责任。\
        督促指导电商平台报告其自营业务产生的塑料袋等一次性塑料制品的使用、回收情况，通过制定平台规则、服务协议，开展宣传推广等措施，引导平台内经营者减少、替代一次性塑料制品使用，并向社会发布执行情况。\
        指导电商平台企业定期对平台内经营者一次性塑料制品使用、回收情况开展调查，按要求上报评估情况。\
        鼓励电商平台积极对接绿色包装供应商，为平台内经营者提供网络集采服务，推广可循环、易回收、可降解替代产品。 扩大绿色产品销售。\
        支持电商平台扩大节能、环保、绿色等产品销售，设立绿色产品销售专区，加强集中展示和宣传。积极发掘当地绿色优质农产品，主动对接电商平台，扩大绿色农产品，尤其是“三品一标”认证农产品网上销售。\
        引导电商平台挖掘绿色消费需求，与品牌商合作对产品和包装开展绿色设计，打造绿色品牌。 积极引导绿色消费。鼓励电商平台建立积分奖励、信用评分等机制，引导消费者购买绿色商品、使用绿色包装或减量包装，参与包装回收。\
        引导电商平台开展以旧换新等绿色环保主题促销和教育宣传活动，传播绿色环保理念，增加消费者对绿色产品的认知。\
        鼓励电商平台开展绿色公益活动，提升消费者绿色环保意识。 \
        保障措施 通知要求，强化组织领导。\
        各地方商务主管部门要提高政治站位，增强做好电子商务领域绿色发展工作的责任感，切实履行好绿色发展行业管理责任，将该项工作纳入发展规划及年度工作计划，建立健全工作机制，加强与工信、财政、生态环境、邮政管理、市场监管等部门协调联动和信息共享。\
        坚持问题导向和目标导向，结合当地实际，细化任务措施，抓好组织实施，推动落实企业主体责任，支持企业绿色创新。 \
        加大政策支持。综合运用规划、标准、资金、投融资等政策导向，推动电子商务企业绿色发展。\
        建立健全绿色电商评价指标，通过示范创建、综合评估等工作，培育一批绿色电商企业，形成一批可复制推广的环保技术应用、快递包装减量化循环化推广新模式。\
        指导电商企业积极参与财政资金支持的绿色技术研发项目，鼓励符合条件的电商企业在开展绿色、可循环快递包装规模化应用中申请绿色信贷。 \
        加强宣传引导。及时总结电商企业绿色发展成效和典型做法，报送优秀典型和案例，利用全国电子商务公共服务平台以及报纸、广播电视、网络新媒体等渠道加强宣传推广。\
        充分调动行业协会、服务机构等中介组织的积极性，加强法规、标准宣贯，组织更多企业响应《电子商务绿色发展倡议》，推动行业自律，更好履行社会责任。\
        利用“全国节能宣传周”等节点，积极宣传绿色消费理念，引导全社会形成简约适度、绿色低碳的生活方式。"
    text = text.strip().replace(" ", '')
    tr = MMR(return_importance=True)
    sentences = tr.summarize_document(text, k=3, mmr_lambda=0.9)
    print(sentences)