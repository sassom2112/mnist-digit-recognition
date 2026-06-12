FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --upgrade pip --quiet && \
    pip install -r requirements.txt --quiet

COPY src/app.py ${LAMBDA_TASK_ROOT}/

CMD ["app.lambda_handler"]
