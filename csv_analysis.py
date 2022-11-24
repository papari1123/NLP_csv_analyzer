import os
import pickle

import numpy as np
import pandas as pd
import torch
from torchmetrics import PearsonCorrCoef

from plotly.subplots import make_subplots
import plotly.express as px 
import plotly.graph_objects as go

from pathlib import Path
from natsort import natsorted
from typing import Tuple, List

MULTI_CLF_TASK = 'multi_clf'

class NLPAnalyzer():
    def __init__(self, task=MULTI_CLF_TASK, dict_label_to_num_path=None, tokenizer=None):
        """_summary_

        Args:
            tokenizer (any): 토크나이저, Transformer의 AutoTokenizer를 사용할 수 있음.
            task (_type_, optional): 분석할 데이터와 관련된 task
            dict_label_to_num_path (_type_, optional): 
            classification 문제의 경우, string label을 num값으로 바꿔주기 위한 dictionary pickle 경로
        """
        if task == MULTI_CLF_TASK:
            self.task = MULTI_CLF_TASK
            if dict_label_to_num_path is None:
                raise Exception(f'multi class classification은 dict_label_to_num을 포함해야 합니다.')
        else:
            raise Exception(f'{task}는 지원하지 않습니다.')
        self.tokenizer = tokenizer
        self.dict_label_to_num_path = dict_label_to_num_path
        
        with open(self.dict_label_to_num_path, 'rb') as f:
            dict_label_to_num = pickle.load(f)
            self.label_num = len(dict_label_to_num)

        self.df_dict = dict()  
        self.sample_df_dict = dict() 

        self.anno_label = None
        self.anno_pred = None
        self.anno_sentence = [] 

        # list 성분은 키임.
        self.enabled_df_name_list = []
        self.aux_feature = []
        
    def enable_df_all(self):
        """_summary_
        저장된 모든 dataframe의 결과를 확인할 수 있도록 enable한다.
        """
        self.enabled_df_name_list = list(self.df_dict.keys())
        
    def enable_df_only(self, df_name_list: List[str]):
        """_summary_
        지정한 dataframe의 결과만을 확인하도록 설정한다.
        Args:
            df_name_list (list): 지정할 dataframe의 name list. 
            지정할 dataframe은 put 또는 puts 메소드를 사용해 NLPAnalyzer 인스턴스에 저장된 상태여야 한다.
        """
        self.enabled_df_name_list = df_name_list

    def print_enable_df_list(self):
        """_summary_
        현재 결과를 확인하기로 세팅된 dataframe의 name list 출력.
        """
        print(self.enabled_df_name_list)
        
    def print_df_list(self):
        """_summary_
        NLPAnalyzer 인스턴스에 저장된 모든 dataframe의 name list 출력.
        """
        print(self.df_dict.keys())
        
    def print_columns(self):
        """_summary_
        NLPAnalyzer에 저장된 dataframe들의 컬럼명을 모두 확인한다.
        """
        for df_name in self.enabled_df_name_list:
            print(self.df_dict[df_name].columns)
            
    def describe_all(self, show_cols=None):
        """_summary_
        enabled dataframe들의 feature에 대한 통계를 모두 출력한다.
        Args:
            show_cols (_type_, optional): 확인할 feature들을 정의. None일 경우 모두 확인.
        """
        for df_name in self.enabled_df_name_list:
            df = self.df_dict[df_name]
            if show_cols:
                print(df_name, df[show_cols].describe())
            else:
                print(df_name, df.describe())
    
    def label_to_num(self, label: List[str]) -> List[int]:
        """_summary_
        string label list를 number로 변환한다. label_to_num 딕셔너리를 세팅했을 때만 가능.
        Args:
            label (List[int]): string label list

        Returns:
            List[int]: num label list
        """
        num_label = []
        with open(self.dict_label_to_num_path, 'rb') as f:
            dict_label_to_num = pickle.load(f)
        for v in label:
            num_label.append(dict_label_to_num[v])
        return num_label
    
    def num_to_label(self, num: List[int]) -> List[str]:
        """_summary_
        num label list를 string label list로 변환한다. label_to_num 딕셔너리를 세팅했을 때만 가능.
        Args:
            num (List[int]): num label list

        Returns:
            List[str]: string label list
        """
        str_label = []
        with open(self.dict_label_to_num_path, 'rb') as f:
            dict_num_to_label = {v: k for k, v in pickle.load(f).items()}
        for v in num:
            str_label.append(dict_num_to_label[v])
        return str_label 
    
    def annotate_feature(self, col:str, type:str):
        """_summary_
        csv의 feature에 대해 type을 설정한다. 이후 feature_engineering에서 auxilnary feature를 구할 때 활용된다.
        Args:
            feature (str): type을 설정할 feature의 이름
            type (str): 설정할 type. 현재 'label', 'pred', 'sentence'만 가능.
        """
        if type == 'label':
            self.anno_label = col
        elif type == 'pred':
            self.anno_pred = col 
        elif type == 'sentence':
            self.anno_sentence.append(col)     
    
    def feature_engineering(self, df: pd.DataFrame) ->  pd.DataFrame:
        """_summary_ 
        분석을 위해 auxilary feature를 추가한 dataframe을 반환한다.
        Args:
            df (pd.DataFrame): 기존 dataframe

        Returns:
            pd.DataFrame: auxilary feature가 추가된 dataframe
        """
        
        # 라벨 (number)
        if self.anno_label:
            df['str_label'] = df[self.anno_label]
            df['num_label'] = self.label_to_num(df['str_label'])          
            
        
        if self.anno_pred and self.anno_label:    
            labels_probs = np.zeros((len(df), self.label_num))  
            labels_confidence = np.zeros((len(df)))
            preds = np.zeros((len(df)))         
            for idx, row in df.iterrows():
                prob_str = row[self.anno_pred]
                labels_probs[idx] = np.array(prob_str.split())
                # TODO : 간결하게 줄이는 법?
                labels_confidence[idx] = labels_probs[idx, df['num_label'].iloc[idx]]
            preds = np.argmax(labels_probs, axis = -1)
            
            # 추가 feature
            df['num_pred'] = pd.Series(preds) # number 표현된 예측값
            df['pred'] = self.num_to_label(df['num_pred']) # str 표현된 예측값
            df['pred.confidence'] = np.max(labels_probs) # 예측값에 대한 confidence
            df['label.confidence'] = labels_confidence # 라벨값에 대한 confidence
            df['answer'] = np.where(df['pred'] == df['label'], 1, 0) # 정답 여부

        for name in self.anno_sentence:
            # 공백 개수
            df[name + '.space'] = [s.count(' ') for s in df[name]]
            # 문장 길이
            df[name + '.len'] = [len(s) for s in df[name]]
            # 토큰 개수
            if self.tokenizer:
                df[name + '.token_num'] = [len(self.tokenizer.encode(s)) for s in df[name]]
            else:
                print('tokenizer가 없으므로, token num은 계산되지 않음.')
        return df 
        
    def puts(self, dir_path: str, aux_feature=True):
        """_summary_
        디렉토리 내에 csv파일들을 모두 NLPAnalyzer에 추가한다.
        추가된 csv파일들은 모두 enable 상태임.
        Args:
            dir_path (str): 추가할 csv파일이 포함된 폴더 경로
            aux_feature (boolean): Analyzer에 설정된 auxinary feature를 추가함.
        """
        for p in natsorted(Path(dir_path).glob('*.csv'), key=str):
            self.put(p, aux_feature)

    def put(self, path: None, aux_feature=True):
        """_summary_
        csv파일을 읽고 설정된 auxilary feauture를 포함해 NLPAnalyzer에 추가한다.
        추가된 csv파일은 모두 enable 상태임.
        Args:
            path (Path or str): 추가할 csv파일의 경로
            aux_feature (boolean): Analyzer에 설정된 auxinary feature를 추가함.
        """
        df = pd.read_csv(path)
        print(f'put {path}..')
        name = path.name[:-4]
        if name in self.df_dict:
            print(f'{name} already exists. it will be overwritten.')
        if aux_feature:
            self.df_dict[name] = self.feature_engineering(df)
        else:
            self.df_dict[name] = df
        self.enabled_df_name_list.append(name)
        
    def gets(self, name_list=None) -> dict:
        """_summary_
        name_list의 name을 가진 dataframe을 반환함. name_list가 없는 경우, 모두 반환.
        Args:
            name_list (_type_, optional): dataframe의 이름 list.

        Returns:
            dict: dataframe이 담긴 딕셔너리
        """
        if name_list == None:
            return self.df_dict
        else:
            return {name: self.df_dict[name] for name in name_list}

    def get(self, name:str) -> pd.DataFrame:
        """_summary_
        1개의 dataframe을 반환함.
        Args:
            name (str): dataframe 이름

        Returns:
            pd.DataFrame: dataframe
        """
        return self.df_dict[name]

    def sample_engineering(self, df, condition_col, show_cols,  th_list=None):
        """_summary_
        통계량을 게산해서 dataframe을 반환.
        ※ 검증안된 함수임. 수정중.
        Args:
            df (_type_): _description_
            condition_col (_type_): _description_
            show_cols (_type_): _description_
            th_list (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        index_list, th_list = self.get_condition_cat_index(df, condition_col, th_list)
        
        sfd = pd.DataFrame()
        for idx, th in zip(index_list, [f'~{th_list[0]}'] + th_list):
            if df[:][idx].shape[0] == 0:
                print(f'{th} : no data.')
                continue
            row = dict()
            row['count'] = df[:][idx].shape[0]
            for col in show_cols:
                if not isinstance(col, tuple):
                    # 일변량 통계값
                    row[col + '_mean'] = [df[col][idx].mean()]
                    row[col + '_std'] = [df[col][idx].std()]
                else:
                    # 다변량(2) 통계값
                    a, b = col
                    pearson = PearsonCorrCoef()
                    row[f'{a}_{b}_pearson'] = float(pearson(torch.tensor(df[a][idx].values), 
                                                    torch.tensor(df[b][idx].values)))     
            # condition_col 카테고리로 보면 됨.
            row[condition_col] = th
            sfd = pd.concat([sfd, pd.DataFrame(row)], ignore_index=True)

        return sfd

    def show_scatter_plot(self, xyz_col, size_col=None, hover_col_list=None, size=(800, 800)):
        x_col, y_col, z_col = xyz_col
        for df_name in self.enabled_df_name_list:
            df = self.df_dict[df_name]
            fig = px.scatter(df, 
                            x=x_col,
                            y=y_col,
                            color=z_col,
                            size=size_col,
                            hover_data=hover_col_list,
                            width=size[0],
                            height=size[1],
                            color_continuous_scale=px.colors.sequential.Jet,
                            title = df_name,
                            
                            
            )
            fig.show()
        
    def get_condition_cat_index(self, df, condition_col, th_list=None, percentile_mode=False):
        if th_list == None:
            # low outlier, low outlier~25%, 25~75%, 75~high outlier, high outlier~
            th_list = [0, 0, 0, 0]
            th_list[1] = df[condition_col].quantile(0.25)
            th_list[2] = df[condition_col].quantile(0.75)
            IQR = th_list[2] - th_list[1]
            # NOTE : BoxPlot의 outlier는 IQR*1.5를 더하거나 뺀 값보다 큰 최소, 최대값이 TH이 되나, 본 코드에서는 편의상 생략.
            th_list[3] = th_list[2] + IQR*1.5
            th_list[0] = th_list[1] - IQR*1.5
        elif percentile_mode:
            th_list = list(map(lambda x: df[condition_col].quantile(x*100), th_list))
                

        index = [0]*(len(th_list)+1)
        for i in range(len(th_list)+1):
            if i == 0:
                index[i] = df[condition_col] < th_list[i] 
            elif i == len(th_list):
                index[i] = df[condition_col] >= th_list[i-1]
            else:
                index[i] = (df[condition_col] >= th_list[i-1]) & (df[condition_col] < th_list[i])
        
        return index, th_list

    def show_histogram(self, col):
        for df_name in self.enabled_df_name_list:
            df = self.df_dict[df_name]
            fig = px.histogram(df, x=col)
            fig.show()

    def show_bar_plot(self, xy_col, text_col=None):
        if text_col == None:
            text_col = xy_col[1]
        for df_name in self.enabled_df_name_list:
            df = self.df_dict[df_name]
            fig = px.bar(df, x=xy_col[0], y=xy_col[1], text=text_col)
            fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig.update_xaxes(tickangle=45)
            fig.show()
        
    # doc 미완성
    def show_group_stat_bar_plot(self, xy_col, mode = 'mean'):
        """_summary_
        doc 미완성.
        xy_col[0]는 num_label 또는 num_pred만 가능.       
        xy_col[1]은 평균을 낼 수 있는 feature만 가능.       
        mode는 'mean', 'std', 'count 가능.
        count 모드의 경우 xy_col[1]는 무시됨
        """
        th_list = list(range(1, 30))
        subtitle = self.num_to_label([0] + th_list)
        fig = go.Figure()
        for df_name in self.enabled_df_name_list:
            df = self.df_dict[df_name]
            sdf = self.sample_engineering(df=df, condition_col=xy_col[0], show_cols=[xy_col[1]], th_list=th_list)
            if mode == 'mean' or mode == 'std':
                _y = sdf[f'{xy_col[1]}_{mode}']
            elif mode == 'count':
                _y = sdf[f'count']
            fig.add_trace(go.Bar(
                        x=subtitle,
                        y=_y,
                        text=_y,
                        name=df_name
                        ))
        if mode == 'count':
            fig.update_layout(title=f'condtion:{xy_col[0]} plot. [mode:{mode}]', barmode='group', xaxis_tickangle=-45)
        else:
            fig.update_layout(title=f'{xy_col[1]} plot. [condtion:{xy_col[0]}, mode:{mode}]', barmode='group', xaxis_tickangle=-45)
        fig.update_xaxes(tickangle=45)
        fig.show()

    def show_condition_dist_plot(self, condition_col, show_col, violin_mode=False, 
                        th_list=None, subtitle=None, size=(500, 1000)):
        for df_name in self.enabled_df_name_list:
            df = self.df_dict[df_name]
            index, th_list = self.get_condition_cat_index(df, condition_col, th_list)
            if not subtitle:
                subtitle = [f'~{th_list[0]:.2f}'] + list(map(lambda x: f'{x:.2f}', th_list))
            
            fig = make_subplots(
                rows=1, cols=len(th_list)+1, shared_yaxes=True)

            if violin_mode:
                for i in range(len(th_list)+1):
                    fig.add_trace(go.Violin(y = df[show_col][index[i]], name=subtitle[i], points='all'), 
                                    row=1, col=i+1)
            else:
                for i in range(len(th_list)+1):
                    fig.add_trace(go.Box(y = df[show_col][index[i]], name=subtitle[i]), row=1, col=i+1)

            fig.update_layout(height=size[0], width=size[1],
                            title_text= f"y={show_col}, condition:{condition_col}")
            fig.update_xaxes(tickangle=45)
            fig.show()

    def show_group_dist_plot(self, show_col, violin_mode=False, size=(500, 1000)):
        df_list = list(self.df_dict.values())
        subtitle = list(self.df_dict.keys())
        
        fig = make_subplots(
            rows=1, cols=len(df_list), shared_yaxes=True)
        if violin_mode:
            for i, fd in enumerate(df_list):
                fig.add_trace(go.Violin(y = fd[show_col], name=subtitle[i], points='all', 
                                        ), row=1, col=i+1)
        else:
            for i, fd in enumerate(df_list):
                fig.add_trace(go.Box(y = fd[show_col], name=subtitle[i]), row=1, col=i+1)

        fig.update_layout(height=size[0], width=size[1],
                        title_text= f"y={show_col}")
        fig.update_xaxes(tickangle=45)
        fig.show()

    def show_condition_group_dist_plot(self, condition_col, show_col, th_list, 
                                    violin_mode=False, ref_th_title = None, size = (500, 1000)): 
        df_list = list(self.df_dict.values())
        subtitle = list(self.df_dict.keys())
        # 여러 개의 모델을 함께 비교해야 하므로, TH_LIST는 고정되어야 함.
        if not ref_th_title:    
            ref_th_title = [f'~{th_list[0]:.2f}'] + list(map(lambda x: f'{x:.2f}', th_list))
        
        for th_idx in range(len(th_list)+1):
            fig = make_subplots(
            rows=1, cols=len(df_list), subplot_titles=None, shared_yaxes=True)
            
            if violin_mode:
                for i, fd in enumerate(df_list):
                    index, _ = self.get_condition_cat_index(df_list[i], condition_col, th_list)
                    fig.add_trace(go.Violin(y = fd[show_col][index[th_idx]], points='all', name=subtitle[i]
                                            ), row=1, col=i+1)
            else:
                for i, fd in enumerate(df_list):
                    index, _ = self.get_condition_cat_index(df_list[i], condition_col, th_list)
                    fig.add_trace(go.Box(y = fd[show_col][index[th_idx]], name=subtitle[i]), row=1, col=i+1)
                    
            fig.update_layout(height=size[0], width=size[1], 
                            title_text= f'{condition_col} {ref_th_title[th_idx]}:y={show_col}' )          
            fig.update_xaxes(tickangle=45) 
            fig.show()